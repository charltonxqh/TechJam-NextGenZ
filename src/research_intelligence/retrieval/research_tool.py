"""
Description: Provides general web, webpage, and academic search capabilities for autonomous ML research.
Owner: Charlton / David
Input: Search query or webpage URL
Output: Structured search results or cleaned webpage content
"""

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import requests

from bs4 import BeautifulSoup

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


class MLResearchTool:

    def __init__(
        self,
    ) -> None:

        self.ddgs = DDGS()

        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }

    def search_web(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict]:
        """
        Search the public web for papers, repositories,
        documentation, competition solutions, technical
        articles, and other potentially useful resources.
        """

        try:

            results = self.ddgs.text(
                query,
                max_results=max_results,
            )

            structured_results = []

            for result in results:

                structured_results.append(
                    {
                        "source_type": "web",
                        "title": result.get(
                            "title",
                            "",
                        ),
                        "url": result.get(
                            "href",
                            "",
                        ),
                        "snippet": result.get(
                            "body",
                            "",
                        ),
                    }
                )

            return structured_results

        except Exception as error:

            return [
                {
                    "source_type": "error",
                    "title": "",
                    "url": "",
                    "snippet": "",
                    "error": (
                        f"Web search failed: "
                        f"{error}"
                    ),
                }
            ]

    def read_page(
        self,
        url: str,
        max_chars: int = 8000,
    ) -> dict:
        """
        Fetch one webpage and return cleaned readable text.
        """

        try:

            response = requests.get(
                url,
                headers=self.headers,
                timeout=10,
            )

            response.raise_for_status()

            content_type = (
                response.headers
                .get(
                    "Content-Type",
                    "",
                )
                .lower()
            )

            if (
                "text/html"
                not in content_type
                and "text/plain"
                not in content_type
            ):

                return {
                    "source_type": "error",
                    "title": "",
                    "url": url,
                    "content": "",
                    "error": (
                        "Unsupported webpage "
                        f"content type: "
                        f"{content_type}"
                    ),
                }

            soup = BeautifulSoup(
                response.text,
                "html.parser",
            )

            for element in soup(
                [
                    "script",
                    "style",
                    "nav",
                    "footer",
                    "aside",
                ]
            ):

                element.extract()

            text = soup.get_text(
                separator=" ",
                strip=True,
            )

            if len(
                text
            ) > max_chars:

                text = (
                    text[:max_chars]
                    + "\n"
                    + "...[Content truncated]..."
                )

            title = ""

            if (
                soup.title
                and soup.title.string
            ):

                title = (
                    soup.title.string
                    .strip()
                )

            return {
                "source_type": "webpage",
                "title": title,
                "url": url,
                "content": text,
            }

        except Exception as error:

            return {
                "source_type": "error",
                "title": "",
                "url": url,
                "content": "",
                "error": (
                    f"Failed to read page: "
                    f"{error}"
                ),
            }

    def search_arxiv(
        self,
        query: str,
        max_results: int = 3,
    ) -> list[dict]:
        """
        Search arXiv for academic papers relevant to the
        current research question.
        """

        try:

            encoded_query = (
                urllib.parse.quote(
                    query
                )
            )

            url = (
                "https://export.arxiv.org/"
                "api/query"
                f"?search_query=all:"
                f"{encoded_query}"
                "&start=0"
                f"&max_results="
                f"{max_results}"
            )

            request = (
                urllib.request.Request(
                    url,
                    headers=self.headers,
                )
            )

            with urllib.request.urlopen(
                request,
                timeout=10,
            ) as response:

                xml_data = (
                    response.read()
                )

            root = ET.fromstring(
                xml_data
            )

            namespaces = {
                "atom": (
                    "http://www.w3.org/"
                    "2005/Atom"
                )
            }

            papers = []

            for entry in root.findall(
                "atom:entry",
                namespaces,
            ):

                title_node = entry.find(
                    "atom:title",
                    namespaces,
                )

                summary_node = entry.find(
                    "atom:summary",
                    namespaces,
                )

                id_node = entry.find(
                    "atom:id",
                    namespaces,
                )

                if (
                    title_node is None
                    or summary_node is None
                    or id_node is None
                ):

                    continue

                authors = []

                for author in entry.findall(
                    "atom:author",
                    namespaces,
                ):

                    name_node = author.find(
                        "atom:name",
                        namespaces,
                    )

                    if (
                        name_node is not None
                        and name_node.text
                    ):

                        authors.append(
                            name_node.text.strip()
                        )

                published_node = entry.find(
                    "atom:published",
                    namespaces,
                )

                papers.append(
                    {
                        "source_type": "paper",
                        "title": (
                            title_node.text
                            .strip()
                            .replace(
                                "\n",
                                " ",
                            )
                        ),
                        "authors": authors,
                        "url": (
                            id_node.text
                            .strip()
                        ),
                        "summary": (
                            summary_node.text
                            .strip()
                            .replace(
                                "\n",
                                " ",
                            )
                        ),
                        "published": (
                            published_node.text
                            .strip()
                            if (
                                published_node
                                is not None
                                and published_node.text
                            )
                            else None
                        ),
                    }
                )

            return papers

        except Exception as error:

            return [
                {
                    "source_type": "error",
                    "title": "",
                    "authors": [],
                    "url": "",
                    "summary": "",
                    "error": (
                        f"arXiv search failed: "
                        f"{error}"
                    ),
                }
            ]
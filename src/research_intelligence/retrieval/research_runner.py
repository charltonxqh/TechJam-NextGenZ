"""
Description: Executes one autonomous online research action, records its retrieval trace, extracts evidence, and stores useful research knowledge.
Owner: Charlton / David
Input: Research query, source preference, task context, iteration, and knowledge store
Output: Number of useful research evidence items stored
"""

from src.research_intelligence.knowledge_store import (
    ResearchKnowledgeStore,
)

from src.research_intelligence.retrieval.evidence_extractor import (
    EvidenceExtractor,
)

from src.research_intelligence.retrieval.research_tool import (
    MLResearchTool,
)

from src.research_intelligence.retrieval.research_trace import (
    ResearchTrace,
)


class ResearchRunner:

    def __init__(
        self,
        research_tool: MLResearchTool,
        evidence_extractor: EvidenceExtractor,
        knowledge_store: ResearchKnowledgeStore,
    ) -> None:

        self.research_tool = (
            research_tool
        )

        self.evidence_extractor = (
            evidence_extractor
        )

        self.knowledge_store = (
            knowledge_store
        )

        self.trace = (
            ResearchTrace(
                path=str(
                    self.knowledge_store
                    .path
                    .with_name(
                        "research_trace.jsonl"
                    )
                )
            )
        )

    def run(
        self,
        query: str,
        source: str,
        task_context: str,
        iteration: int,
        research_action_index: int,
    ) -> int:

        self.trace.record(
            event_type=(
                "research_action_started"
            ),
            data={
                "iteration": iteration,
                "research_action_index": (
                    research_action_index
                ),
                "query": query,
                "source": source,
            },
        )

        sources = []

        # =====================================================
        # 1. General web search
        # =====================================================

        if source in (
            "web",
            "both",
        ):

            print(
                "Searching web..."
            )

            web_results = (
                self.research_tool
                .search_web(
                    query=query,
                    max_results=5,
                )
            )

            valid_web_results = [
                result
                for result
                in web_results
                if (
                    result.get(
                        "source_type"
                    )
                    != "error"
                )
            ]

            web_errors = [
                result
                for result
                in web_results
                if (
                    result.get(
                        "source_type"
                    )
                    == "error"
                )
            ]

            print(
                f"Web results found: "
                f"{len(valid_web_results)}"
            )

            self.trace.record(
                event_type=(
                    "web_search_completed"
                ),
                data={
                    "iteration": iteration,
                    "research_action_index": (
                        research_action_index
                    ),
                    "query": query,
                    "results_found": (
                        len(
                            valid_web_results
                        )
                    ),
                    "errors": [
                        item.get(
                            "error",
                            "",
                        )
                        for item
                        in web_errors
                    ],
                    "results": [
                        {
                            "title": (
                                item.get(
                                    "title",
                                    "",
                                )
                            ),
                            "url": (
                                item.get(
                                    "url",
                                    "",
                                )
                            ),
                            "snippet": (
                                item.get(
                                    "snippet",
                                    "",
                                )
                            ),
                        }
                        for item
                        in valid_web_results
                    ],
                },
            )

            for result_index, result in enumerate(
                valid_web_results[:3]
            ):

                url = (
                    result.get(
                        "url"
                    )
                )

                if not url:

                    sources.append(
                        result
                    )

                    continue

                print(
                    f"Reading web result "
                    f"{result_index + 1}: "
                    f"{result.get('title', '')}"
                )

                page = (
                    self.research_tool
                    .read_page(
                        url
                    )
                )

                if (
                    page.get(
                        "source_type"
                    )
                    == "error"
                ):

                    error = (
                        page.get(
                            "error",
                            "Unknown page-read error.",
                        )
                    )

                    print(
                        f"Page read failed: "
                        f"{error}"
                    )

                    self.trace.record(
                        event_type=(
                            "page_read_failed"
                        ),
                        data={
                            "iteration": (
                                iteration
                            ),
                            "research_action_index": (
                                research_action_index
                            ),
                            "title": (
                                result.get(
                                    "title",
                                    "",
                                )
                            ),
                            "url": url,
                            "error": error,
                        },
                    )

                    if (
                        result.get(
                            "snippet"
                        )
                    ):

                        sources.append(
                            result
                        )

                    continue

                if not (
                    page.get(
                        "title"
                    )
                ):

                    page[
                        "title"
                    ] = (
                        result.get(
                            "title",
                            "",
                        )
                    )

                print(
                    "Page read: success"
                )

                self.trace.record(
                    event_type=(
                        "page_read_completed"
                    ),
                    data={
                        "iteration": iteration,
                        "research_action_index": (
                            research_action_index
                        ),
                        "title": (
                            page.get(
                                "title",
                                "",
                            )
                        ),
                        "url": url,
                        "content_chars": len(
                            page.get(
                                "content",
                                "",
                            )
                        ),
                    },
                )

                sources.append(
                    page
                )

        # =====================================================
        # 2. arXiv search
        # =====================================================

        if source in (
            "arxiv",
            "both",
        ):

            print(
                "Searching arXiv..."
            )

            papers = (
                self.research_tool
                .search_arxiv(
                    query=query,
                    max_results=3,
                )
            )

            valid_papers = [
                paper
                for paper
                in papers
                if (
                    paper.get(
                        "source_type"
                    )
                    != "error"
                )
            ]

            paper_errors = [
                paper
                for paper
                in papers
                if (
                    paper.get(
                        "source_type"
                    )
                    == "error"
                )
            ]

            print(
                f"arXiv papers found: "
                f"{len(valid_papers)}"
            )

            self.trace.record(
                event_type=(
                    "arxiv_search_completed"
                ),
                data={
                    "iteration": iteration,
                    "research_action_index": (
                        research_action_index
                    ),
                    "query": query,
                    "papers_found": (
                        len(
                            valid_papers
                        )
                    ),
                    "errors": [
                        item.get(
                            "error",
                            "",
                        )
                        for item
                        in paper_errors
                    ],
                    "papers": [
                        {
                            "title": (
                                item.get(
                                    "title",
                                    "",
                                )
                            ),
                            "authors": (
                                item.get(
                                    "authors",
                                    [],
                                )
                            ),
                            "url": (
                                item.get(
                                    "url",
                                    "",
                                )
                            ),
                            "published": (
                                item.get(
                                    "published"
                                )
                            ),
                            "summary": (
                                item.get(
                                    "summary",
                                    "",
                                )
                            ),
                        }
                        for item
                        in valid_papers
                    ],
                },
            )

            sources.extend(
                valid_papers
            )

        # =====================================================
        # 3. Remove duplicate URLs
        # =====================================================

        unique_sources = []

        seen_urls = set()

        for research_source in sources:

            url = (
                research_source.get(
                    "url",
                    "",
                )
            )

            if (
                url
                and url in seen_urls
            ):

                continue

            if url:

                seen_urls.add(
                    url
                )

            unique_sources.append(
                research_source
            )

        print(
            f"Sources selected for "
            f"evidence extraction: "
            f"{len(unique_sources)}"
        )

        self.trace.record(
            event_type=(
                "evidence_extraction_started"
            ),
            data={
                "iteration": iteration,
                "research_action_index": (
                    research_action_index
                ),
                "source_count": (
                    len(
                        unique_sources
                    )
                ),
            },
        )

        # =====================================================
        # 4. Extract and store evidence
        # =====================================================

        added = 0

        extraction_failures = 0

        for index, research_source in enumerate(
            unique_sources
        ):

            evidence_id = (
                f"research_"
                f"{iteration}_"
                f"{research_action_index:02d}_"
                f"{index:03d}"
            )

            title = (
                research_source.get(
                    "title",
                    "",
                )
            )

            url = (
                research_source.get(
                    "url",
                    "",
                )
            )

            print(
                f"Extracting evidence "
                f"{index + 1}/"
                f"{len(unique_sources)}: "
                f"{title}"
            )

            try:

                evidence = (
                    self.evidence_extractor
                    .extract(
                        evidence_id=(
                            evidence_id
                        ),
                        source=(
                            research_source
                        ),
                        task_context=(
                            task_context
                        ),
                    )
                )

            except Exception as error:

                extraction_failures += 1

                print(
                    f"Evidence extraction "
                    f"failed: "
                    f"{type(error).__name__}: "
                    f"{error}"
                )

                self.trace.record(
                    event_type=(
                        "evidence_extraction_failed"
                    ),
                    data={
                        "iteration": iteration,
                        "research_action_index": (
                            research_action_index
                        ),
                        "evidence_id": (
                            evidence_id
                        ),
                        "title": title,
                        "url": url,
                        "error_type": (
                            type(
                                error
                            ).__name__
                        ),
                        "error": (
                            str(
                                error
                            )
                        ),
                    },
                )

                continue

            try:

                stored = (
                    self.knowledge_store
                    .add_research_evidence(
                        evidence=(
                            evidence
                        ),
                        iteration=(
                            iteration
                        ),
                    )
                )

            except Exception as error:

                print(
                    f"Evidence storage "
                    f"failed: "
                    f"{type(error).__name__}: "
                    f"{error}"
                )

                self.trace.record(
                    event_type=(
                        "evidence_storage_failed"
                    ),
                    data={
                        "iteration": iteration,
                        "research_action_index": (
                            research_action_index
                        ),
                        "evidence_id": (
                            evidence_id
                        ),
                        "title": (
                            evidence.title
                        ),
                        "url": (
                            evidence.url
                        ),
                        "error_type": (
                            type(
                                error
                            ).__name__
                        ),
                        "error": (
                            str(
                                error
                            )
                        ),
                    },
                )

                continue

            if stored:

                added += 1

                print(
                    f"Stored research evidence: "
                    f"{evidence_id}"
                )

                self.trace.record(
                    event_type=(
                        "evidence_stored"
                    ),
                    data={
                        "iteration": iteration,
                        "research_action_index": (
                            research_action_index
                        ),
                        "evidence_id": (
                            evidence_id
                        ),
                        "source_type": (
                            evidence.source_type
                        ),
                        "title": (
                            evidence.title
                        ),
                        "url": (
                            evidence.url
                        ),
                        "topic": (
                            evidence.topic
                        ),
                        "problem_addressed": (
                            evidence
                            .problem_addressed
                        ),
                        "method": (
                            evidence.method
                        ),
                        "assumptions": (
                            evidence.assumptions
                        ),
                        "relevant_findings": (
                            evidence
                            .relevant_findings
                        ),
                        "applicability": (
                            evidence
                            .applicability
                        ),
                        "implementation_hint": (
                            evidence
                            .implementation_hint
                        ),
                        "confidence": (
                            evidence.confidence
                        ),
                    },
                )

            else:

                print(
                    "Evidence was already "
                    "present in knowledge store."
                )

                self.trace.record(
                    event_type=(
                        "evidence_duplicate"
                    ),
                    data={
                        "iteration": iteration,
                        "research_action_index": (
                            research_action_index
                        ),
                        "evidence_id": (
                            evidence_id
                        ),
                        "title": (
                            evidence.title
                        ),
                        "url": (
                            evidence.url
                        ),
                    },
                )

        # =====================================================
        # 5. Final research-action trace
        # =====================================================

        self.trace.record(
            event_type=(
                "research_action_completed"
            ),
            data={
                "iteration": iteration,
                "research_action_index": (
                    research_action_index
                ),
                "query": query,
                "source": source,
                "sources_considered": (
                    len(
                        unique_sources
                    )
                ),
                "evidence_added": added,
                "extraction_failures": (
                    extraction_failures
                ),
            },
        )

        return added
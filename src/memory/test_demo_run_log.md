# Run Log

## Iteration 1 (model)
**Hypothesis:** Reproduce the official baseline FM to confirm the pipeline works end-to-end.

**Metrics:** GAUC=0.6546000000000001, nDCG@5=0.5346, primary=0.5946

**Code diff:**
```
import numpy as np

class FactorizationMachine:
    def __init__(self, n_features, k=16, lr=0.001):
        self.w0 = 0.0
        self.w = np.zeros(n_features)
        self.V = np.random.normal(0, 0.01, (n_features, k))
        self.lr = lr

    def predict(self, x):
        linear = self.w0 + np.dot(self.w, x)
        interaction = 0.5 * np.sum(
            np.dot(x, self.V) ** 2 - np.dot(x ** 2, self.V ** 2)
        )
        return linear + interaction

    def fit(self, X, y, epochs=5):
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                pred = self.predict(xi)
                error = pred - yi
                self.w0 -= self.lr * error
                self.w -= self.lr * error * xi
```

---

## Iteration 2 (features)
**Hypothesis:** Add user's historical average watch_time as a feature, since long_view likely correlates with past viewing behavior.

**Metrics:** GAUC=0.661, nDCG@5=0.5409999999999999, primary=0.601

**Code diff:**
```
--- previous_iteration
+++ this_iteration
@@ -20,4 +20,11 @@
                 pred = self.predict(xi)
                 error = pred - yi
                 self.w0 -= self.lr * error
-                self.w -= self.lr * error * xi+                self.w -= self.lr * error * xi
+
+
+def add_avg_watch_time_feature(df, train_df):
+    """New: user's historical average watch_time as a feature."""
+    user_avg = train_df.groupby('user_id')['play_time_ms'].mean()
+    df['user_avg_watch_time'] = df['user_id'].map(user_avg).fillna(user_avg.mean())
+    return df
```

---

## Iteration 3 (model)
**Hypothesis:** Replace the linear FM with a DeepFM (embeddings + MLP) to capture nonlinear feature interactions.

**Metrics:** GAUC=0.6648000000000001, nDCG@5=0.5448, primary=0.6048

**Code diff:**
```
--- previous_iteration
+++ this_iteration
@@ -1,30 +1,27 @@
 import numpy as np
+import torch
+import torch.nn as nn
 
-class FactorizationMachine:
-    def __init__(self, n_features, k=16, lr=0.001):
-        self.w0 = 0.0
-        self.w = np.zeros(n_features)
-        self.V = np.random.normal(0, 0.01, (n_features, k))
-        self.lr = lr
+class DeepFM(nn.Module):
+    """Replaces the plain FM with a neural net on top of the same embeddings."""
+    def __init__(self, num_features, embed_dim=16, hidden=64):
+        super().__init__()
+        self.embedding = nn.Embedding(num_features, embed_dim)
+        self.fm_linear = nn.Linear(num_features, 1)
+        self.mlp = nn.Sequential(
+            nn.Linear(embed_dim, hidden), nn.ReLU(),
+            nn.Linear(hidden, hidden), nn.ReLU(),
+            nn.Linear(hidden, 1)
+        )
 
-    def predict(self, x):
-        linear = self.w0 + np.dot(self.w, x)
-        interaction = 0.5 * np.sum(
-            np.dot(x, self.V) ** 2 - np.dot(x ** 2, self.V ** 2)
-        )
-        return linear + interaction
-
-    def fit(self, X, y, epochs=5):
-        for _ in range(epochs):
-            for xi, yi in zip(X, y):
-                pred = self.predict(xi)
-                error = pred - yi
-                self.w0 -= self.lr * error
-                self.w -= self.lr * error * xi
+    def forward(self, x):
+        emb = self.embedding(x)
+        fm_out = self.fm_linear(x.float())
+        deep_out = self.mlp(emb.mean(dim=1))
+        return torch.sigmoid(fm_out + deep_out)
 
 
 def add_avg_watch_time_feature(df, train_df):
-    """New: user's historical average watch_time as a feature."""
     user_avg = train_df.groupby('user_id')['play_time_ms'].mean()
     df['user_avg_watch_time'] = df['user_id'].map(user_avg).fillna(user_avg.mean())
     return df
```

---

## Iteration 4 (model)
**Hypothesis:** Increase embedding dimension from 16 to 64, expecting more capacity to help the model fit richer interactions.

**Metrics:** GAUC=0.6557999999999999, nDCG@5=0.5358, primary=0.5958

**Code diff:**
```
--- previous_iteration
+++ this_iteration
@@ -3,10 +3,9 @@
 import torch.nn as nn
 
 class DeepFM(nn.Module):
-    """Replaces the plain FM with a neural net on top of the same embeddings."""
-    def __init__(self, num_features, embed_dim=16, hidden=64):
+    def __init__(self, num_features, embed_dim=64, hidden=64):
         super().__init__()
-        self.embedding = nn.Embedding(num_features, embed_dim)
+        self.embedding = nn.Embedding(num_features, embed_dim)  # 64, was 16
         self.fm_linear = nn.Linear(num_features, 1)
         self.mlp = nn.Sequential(
             nn.Linear(embed_dim, hidden), nn.ReLU(),

```

---

## Iteration 5 (training)
**Hypothesis:** Revert embedding dimension back to 16, add dropout(0.2) on the MLP layers instead to regularize without losing capacity.

**Metrics:** GAUC=0.6702999999999999, nDCG@5=0.5503, primary=0.6103

**Code diff:**
```
--- previous_iteration
+++ this_iteration
@@ -3,13 +3,13 @@
 import torch.nn as nn
 
 class DeepFM(nn.Module):
-    def __init__(self, num_features, embed_dim=64, hidden=64):
+    def __init__(self, num_features, embed_dim=16, hidden=64):
         super().__init__()
-        self.embedding = nn.Embedding(num_features, embed_dim)  # 64, was 16
+        self.embedding = nn.Embedding(num_features, embed_dim)  # reverted to 16
         self.fm_linear = nn.Linear(num_features, 1)
         self.mlp = nn.Sequential(
-            nn.Linear(embed_dim, hidden), nn.ReLU(),
-            nn.Linear(hidden, hidden), nn.ReLU(),
+            nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
+            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
             nn.Linear(hidden, 1)
         )
 

```

---

# Summary
- Total iterations: 5
- Manual interventions: 0
- Total tokens: 0
- Total wall-clock (sec): 5.0
- Best primary score: 0.6103 (iteration 5)
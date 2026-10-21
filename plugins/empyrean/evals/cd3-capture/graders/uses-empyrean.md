---
type: regex
target: trace
pattern: 'pip install[^\\]{0,80}empyrean|uv (?:add|pip install)[^\\]{0,80}empyrean|import empyrean|from empyrean import'
arm: both
weight: 2
---

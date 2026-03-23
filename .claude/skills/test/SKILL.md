---
name: test
description: Run tests. Use after code changes to validate. Arguments: unit (default, no GPU), e2e (with models), filter name, or all.
disable-model-invocation: true
argument-hint: [unit|e2e|all|FilterName]
allowed-tools: Bash
---

# Test

Run unit and/or E2E tests.

- `unit` (default): Quick tests, no model downloads. CI-safe.
- `e2e`: Full pipeline tests with real models. Requires metallib + cached weights.
- `all`: Both unit and E2E.
- Any other argument: passed as `--filter` to swift test.

```bash
arg="${ARGUMENTS:-unit}"
case "$arg" in
  unit) swift test --skip E2E 2>&1 | tail -20 ;;
  e2e)  swift test 2>&1 | tail -30 ;;
  all)  swift test 2>&1 | tail -30 ;;
  *)    swift test --filter "$arg" 2>&1 | tail -20 ;;
esac
```

E2E test classes MUST be prefixed with `E2E`. Unit test classes must NOT contain `E2E`.

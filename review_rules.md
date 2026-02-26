# Coding Rules — Python / Backend

## Code Style

1. **PEP8 and `wemake-python-styleguide`:** The foundation. wemake includes checks for complexity, clarity, and consistency.

2. **Docstrings (Sphinx style):**
   - Document modules, classes, functions, and methods.
   - Explain *what* the code does and *why*, not *how* (unless it is a complex algorithm).
   - Use `:param:`, `:raises:`, `:return:` annotations.

3. **Line length:** Max 79 characters.

4. **Indentation:** 4 spaces.

5. **Naming:**
   - `snake_case` for variables, functions, methods, modules.
   - `CamelCase` for classes.
   - `UPPER_SNAKE_CASE` for constants.
   - Names must be descriptive. Avoid single-letter names (except `i`, `j`, `k` in loops, `e` in `except`).

6. **Imports:** Follow `isort` rules.

7. **Strings:**
   - Use single quotes (`'`) for strings.
   - Use `.format()` for string formatting. Do NOT use f-strings or old `%`-style.

8. **Comments:**
   - Do NOT use `#` comments for obvious things. Write self-documenting code.
   - `#` comments are acceptable for: complex/non-obvious logic, TODO/FIXME/HACK, temporarily disabled code.
   - Use docstrings (`"""Docstring"""`) for documentation.

9. **Simplicity:**
   - Avoid deep nesting (`if/for/try` more than 2-3 levels). Use guard clauses (early returns).
   - Limit function argument count. If many arguments — consider a Pydantic model.
   - Do not use single-line `if`, `for`, `try/except` if it reduces readability.

10. **Whitespace:** Spaces around binary operators (`=`, `+`, `-`, `==`, `!=`, `in`, `is`, etc.), but not inside parentheses.

## Correctness & Logic

- **Functionality:** Does the code work as expected? Are the main scenarios covered?
- **Edge cases:** Are `None`, empty lists, invalid input, boundary values handled?
- **Error handling:** Are errors handled correctly? Are domain exceptions used? Are important errors not silently swallowed?
- **Concurrency/Async:** Are there race conditions? Is `async/await`, `asyncio.gather`, `async with` used correctly? Are there blocking calls inside async code?

## Architecture & Design

- **SOLID:** Are SOLID principles followed? Especially Single Responsibility and Dependency Inversion.
- **Clean Architecture:** Is logic separated into layers (API, Services, Repositories)? Are dependencies injected? Are abstractions (ports) used?
- **DRY (Don't Repeat Yourself):** Is there code duplication?
- **Pydantic Models:** Are Pydantic models used for API? Are Request/Response/Internal models separated? Is validation used?
- **Configuration:** Are important parameters not hardcoded (connection strings, API keys)? Are env variables / config files used?

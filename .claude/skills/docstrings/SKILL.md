---
name: docstrings
description: >-
  Writes and fixes Python docstrings in Google style with Args, Returns,
  and Raises sections. Use when the user asks to document, add docstrings
  to, or improve docstrings in a Python function, class, or module.
---

When adding or rewriting a docstring:

1. Open with a one-line summary in the imperative mood, ending with a
   period. Write "Return the total.", not "Returns the total."
2. Add `Args:`, `Returns:`, and `Raises:` sections in Google style. Leave
   out any section that doesn't apply.
3. Name the type of every parameter and return value. Skip `self` and
   `cls`.
4. Keep the code unchanged. Only add or replace docstrings.
5. Show the updated code, not a description of the changes.


Google Style:

Sections use a heading followed by an indented block. Names come first,
then the type in parentheses, then a description.

    def restock(items, name, quantity):
        """Add stock for one item and return its new count.

        Args:
            items (dict[str, int]): Current stock, keyed by item name.
            name (str): The item to restock.
            quantity (int): How many units to add. Must be positive.

        Returns:
            int: The item's count after restocking.

        Raises:
            ValueError: If quantity is zero or negative.
        """

Rules:

- Sections, in this order: `Args:`, then `Returns:` (or `Yields:` for
  generators), then `Raises:`.
- Indent section contents by four spaces.
- Describe class attributes in an `Attributes:` section of the class
  docstring, not in `__init__`.
- Google's guide accepts descriptive ("Adds stock.") or imperative
  ("Add stock.") summaries as long as a file is consistent. This skill
  always uses the imperative mood, following PEP 257.
 
"""Shared paginated-list-view utilities used by TelegramCommandProcessor and
aka_no_claw command handlers.

Extracting these allows aka_no_claw to build list views (km / kc / sl / hl …)
without duplicating the markup logic, while maintaining the one-way dependency
direction (aka_no_claw → price_monitor_bot).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

LIST_VIEW_MODE_READ = "r"
LIST_VIEW_MODE_EDIT = "e"
LIST_VIEW_PAGE_SIZE = 5


@dataclass(frozen=True, slots=True)
class ListRow:
    """One row of a paginated list view.

    `id` lands in the delete-button callback_data — keep it short enough
    that ``del:<kind>:<id>`` stays under Telegram's 64-byte limit.
    `text` is the rendered block for read mode; `short_label` is the
    truncated label on the delete button in edit mode.
    `extra_buttons` is appended next to the delete button on the same row.
    `label_button` when set is inserted as a full-width row above the
    action buttons in edit mode.
    """

    id: str
    text: str
    short_label: str
    extra_buttons: tuple[dict[str, object], ...] = field(default_factory=tuple)
    label_button: dict[str, object] | None = None


# Type alias used by view/deleter registries.
ViewFn = Callable[..., tuple[str, "dict[str, object] | None", int]]
DeleterEntry = tuple[Callable[[str], bool], str]  # (fn, human_label)


def build_list_view(
    *,
    list_kind: str,
    items: list[ListRow],
    page: int,
    mode: str,
    list_title: str,
    empty_message: str,
    read_mode_row_buttons: bool = False,
) -> tuple[str, dict[str, object] | None, int]:
    """Render a paginated list view. Returns ``(text, reply_markup, clamped_page)``.

    ``clamped_page`` is ``page`` snapped into ``[0, total_pages)``.

    When ``read_mode_row_buttons`` is set, each visible row's ``extra_buttons``
    (and its ``label_button``, if any) are also emitted in read mode — used by
    views that want a per-row action (e.g. ▶️ play) without entering edit mode.
    """
    if not items:
        return empty_message, None, 0

    total = len(items)
    total_pages = max(1, (total + LIST_VIEW_PAGE_SIZE - 1) // LIST_VIEW_PAGE_SIZE)
    clamped = max(0, min(page, total_pages - 1))
    start = clamped * LIST_VIEW_PAGE_SIZE
    visible = items[start : start + LIST_VIEW_PAGE_SIZE]

    header = f"{list_title}  第 {clamped + 1}/{total_pages} 頁（共 {total} 筆）"
    body_lines = [row.text for row in visible if row.text]
    text = "\n".join([header, "", *body_lines] if body_lines else [header])

    keyboard: list[list[dict[str, object]]] = []
    if mode == LIST_VIEW_MODE_EDIT:
        for row in visible:
            if row.label_button is not None:
                keyboard.append([row.label_button])
            btn_label = f"❌ 刪除 {row.short_label}".strip() if row.short_label else "❌ 刪除"
            row_buttons: list[dict[str, object]] = [{
                "text": btn_label,
                "callback_data": f"del:{list_kind}:{row.id}",
            }]
            row_buttons.extend(row.extra_buttons)
            keyboard.append(row_buttons)
    elif read_mode_row_buttons:
        for row in visible:
            if row.label_button is not None:
                keyboard.append([row.label_button])
            if row.extra_buttons:
                keyboard.append(list(row.extra_buttons))

    nav: list[dict[str, object]] = []
    if clamped > 0:
        nav.append({"text": "⬅️ 上頁", "callback_data": f"pg:{list_kind}:{clamped - 1}:{mode}"})
    if mode == LIST_VIEW_MODE_READ:
        nav.append({"text": "✏️ 編輯", "callback_data": f"pg:{list_kind}:{clamped}:{LIST_VIEW_MODE_EDIT}"})
    else:
        nav.append({"text": "✓ 完成", "callback_data": f"pg:{list_kind}:{clamped}:{LIST_VIEW_MODE_READ}"})
    if clamped < total_pages - 1:
        nav.append({"text": "下頁 ➡️", "callback_data": f"pg:{list_kind}:{clamped + 1}:{mode}"})
    nav.append({"text": "✖️ 關閉", "callback_data": f"close:{list_kind}"})
    keyboard.append(nav)

    return text, {"inline_keyboard": keyboard}, clamped

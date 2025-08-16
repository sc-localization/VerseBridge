# type:ignore
# TODO: replace with existing tools or add typing

"""
Streamlit application for reviewing and correcting Named Entity Recognition (NER) datasets.
Provides a user-friendly interface for filtering, editing, and annotating entities with features
like entity search, pagination, and auto-save functionality.
"""


import sys
import os

# Add parent directory to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import re
import json
import hashlib
from datetime import datetime
import streamlit as st
from annotated_text import annotated_text

from src.config import ConfigManager
from src.utils import AppLogger, FileUtils
from src.type_defs import ArgLoggerType
from typing import List, Dict, Tuple, Set, Optional


class SessionState:
    """Manages Streamlit session state for persistence across reruns."""

    def __init__(self):
        self._initialize_defaults()

    def _initialize_defaults(self) -> None:
        """Sets default session state values if not already present."""
        defaults = {
            "internal_data": None,
            "data_hash": "",
            "save_needed": False,
            "current_page": 1,
            "entity_search_query": "",
            "committed_entity_search_query": "",
            "custom_categories": [],
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def get(self, key: str, default: any = None) -> any:
        """Retrieves a value from session state with an optional default."""
        return st.session_state.get(key, default)

    def set(self, key: str, value: any) -> None:
        """Sets a value in session state."""
        st.session_state[key] = value

    def delete(self, key: str) -> None:
        """Deletes a key from session state if it exists."""
        if key in st.session_state:
            del st.session_state[key]


class CategoryManager:
    """Manages entity categories, including default and custom categories."""

    def __init__(self, config: ConfigManager, logger: ArgLoggerType):
        self.config = config
        self.logger = logger
        self.default_categories = config.ner_config.categories[1:]  # Exclude "ALL"

    @property
    def custom_categories(self) -> List[str]:
        """Returns custom categories from session state."""
        return st.session_state.get("custom_categories", [])

    def get_all(self) -> List[str]:
        """Returns all available categories (default + custom)."""
        return self.default_categories + self.custom_categories

    def add_custom(self, new_category: str) -> bool:
        """Adds a new custom category if it doesn't exist."""
        if not new_category or new_category in self.get_all():
            return False

        st.session_state.setdefault("custom_categories", []).append(new_category)
        self.logger.info(f"Added custom category: {new_category}")

        return True

    def exists(self, category: str) -> bool:
        """Checks if a category exists."""
        return category in self.get_all()


class DataFilter:
    """Handles filtering of dataset examples based on user criteria."""

    def __init__(self, logger: ArgLoggerType):
        self.logger = logger

    def _has_unreviewed(self, example: Dict) -> bool:
        """Checks if an example has unreviewed entities."""
        return any(
            not entity.get("reviewed", False) for entity in example.get("entities", [])
        )

    def _is_fully_reviewed(self, example: Dict) -> bool:
        """Checks if all entities in an example are reviewed."""
        return all(entity.get("reviewed", False) for entity in example.get("entities", [])) if example.get("entities") else False

    def _has_overlaps(self, entities: List[Dict]) -> bool:
        """Checks for overlapping entities."""
        sorted_entities = sorted(entities, key=lambda x: x["start"])

        for i in range(len(sorted_entities) - 1):
            if sorted_entities[i]["end"] > sorted_entities[i + 1]["start"]:
                return True
            
        return False

    def _has_invalid_indices(self, entities: List[Dict], text_length: int) -> bool:
        """Checks for entities with invalid indices."""
        return any(
            not (0 <= e["start"] < e["end"] <= text_length)
            for e in entities
            if "start" in e and "end" in e
        )

    def filter(
        self,
        data: List[Dict],
        selected_category: str,
        show_only_unreviewed: bool,
        show_overlaps: bool,
        show_empty: bool,
        show_invalid: bool,
        show_only_reviewed: bool,
        show_unique_reviewed: bool,
        entity_search_query: str,
        entity_search_type: str = "text",
    ) -> List[Dict]:
        """Filters data based on user-selected criteria."""
        if show_unique_reviewed:
            seen_entities = set()
            filtered_data = []
            
                entities_to_check = tuple(
                    (e["start"], e["end"], e["label"])
                    for e in ex["entities"]
                    if e.get("reviewed", False)
                )
            
                if entities_to_check and entities_to_check not in seen_entities:
                    seen_entities.add(entities_to_check)
                    filtered_data.append(ex)

            return filtered_data

        filtered_data = data
        if selected_category != "ALL":
            filtered_data = [ex for ex in filtered_data if any(e["label"] == selected_category for e in ex["entities"])]

        if entity_search_query.strip():
            query_lower = entity_search_query.lower()

            if entity_search_type == "text":
                filtered_data = [
                    ex
                    for ex in filtered_data
                    if any(
                        ex["text"][e["start"] : e["end"]].lower() == query_lower
                        for e in ex.get("entities", [])
                    )
                ]
            elif entity_search_type == "id":
                filtered_data = [
                    ex
                    for ex in filtered_data
                    if str(ex.get("id", "")).lower() == query_lower
                ]

        result = []

        for ex in filtered_data:
            entities = ex.get("entities", [])
            text_length = len(ex["text"])
            
            if (show_empty and entities) or \
               (show_only_unreviewed and not self._has_unreviewed(ex)) or \
               (show_only_reviewed and not self._is_fully_reviewed(ex)) or \
               (show_overlaps and not self._has_overlaps(entities)) or \
               (show_invalid and not self._has_invalid_indices(entities, text_length)):
                continue
            
            result.append(ex)

        return result

    def get_filter_feedback(self, filtered_data: List[Dict], *filters) -> None:
        """Displays feedback about filtered results."""
        if not filtered_data:
            reasons = []
            (selected_category, show_only_unreviewed, show_overlaps, show_empty, show_invalid, show_only_reviewed, show_unique_reviewed, entity_search_query, entity_search_type) = filters
            
            if entity_search_query.strip():
                reasons.append(f"{entity_search_type} '{entity_search_query}'")
            if selected_category != "ALL":
                reasons.append(f"category '{selected_category}'")
            if show_only_unreviewed:
                reasons.append("unreviewed entities")
            if show_only_reviewed:
                reasons.append("all entities reviewed")
            if show_overlaps:
                reasons.append("overlapping entities")
            if show_empty:
                reasons.append("no entities")
            if show_invalid:
                reasons.append("invalid indices")
            if show_unique_reviewed:
                reasons.append("unique reviewed entities")

            st.info(f"No examples match the selected filters: {', '.join(reasons)}.")
        else:
            (selected_category, show_only_unreviewed, show_overlaps, show_empty, show_invalid, show_only_reviewed, show_unique_reviewed, entity_search_query, entity_search_type) = filters
            st.write(
                f"Filtered {len(filtered_data)} examples (entity: '{entity_search_query}' (type: {entity_search_type}), "
                f"category: {selected_category}, unreviewed: {show_only_unreviewed}, "
                f"empty: {show_empty}, overlaps: {show_overlaps}, invalid: {show_invalid}, "
                f"reviewed: {show_only_reviewed}, unique reviewed: {show_unique_reviewed})"
            )


class DataManager:
    """Manages dataset loading, saving, and hashing."""

    def __init__(
        self, config: ConfigManager, file_utils: FileUtils, logger: ArgLoggerType
    ):
        self.config = config
        self.file_utils = file_utils
        self.logger = logger
        self.session = SessionState()

    def load_data(self) -> List[Dict]:
        """Loads dataset from temp or original path."""
        path = (
            self.config.ner_path_config.temp_streamlit_data_path
            if self.config.ner_path_config.temp_streamlit_data_path.exists()
            else self.config.ner_path_config.extracted_ner_data_path
        )

        self.logger.info(f"Loading data from {path}")

        data = self.file_utils.load_json(path)

        self.session.set("internal_data", data)
        self.session.set("data_hash", self._compute_data_hash(data))

        return data

    def _compute_data_hash(self, data: List[Dict]) -> str:
        """Computes a hash of the dataset for change detection."""
        try:
            data_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
            return hashlib.sha256(data_str.encode("utf-8")).hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to compute data hash: {str(e)}")
            return ""

    def save_temp_data(self) -> bool:
        """Saves temporary data to file."""
        data = self.session.get("internal_data", [])

        if not data:
            self.logger.info("No data to save.")
            return False

        try:
            self.file_utils.save_json(
                data, self.config.ner_path_config.temp_streamlit_data_path
            )
            self.logger.info(
                f"Temporary data saved to {self.config.ner_path_config.temp_streamlit_data_path}"
            )
            self.session.set("data_hash", self._compute_data_hash(data))
            self.session.set("save_needed", False)

            return True
        except Exception as e:
            self.logger.error(f"Failed to save temp data: {str(e)}")
            st.error(f"Error saving temporary data: {str(e)}")

            return False

    def save_final_data(self) -> bool:
        """Saves final cleaned data to file."""
        data = self.session.get("internal_data", [])
        if not data:
            st.error("No data to save.")
            return False

        try:
            cleaned_data = [
                {
                    "id": item["id"],
                    "text": item["text"],
                    "entities": [
                        {"start": e["start"], "end": e["end"], "label": e["label"]}
                        for e in item["entities"]
                        if "start" in e and "end" in e and "label" in e
                    ],
                }
                for item in data
                if item.get("entities")
            ]
            self.file_utils.save_json(
                cleaned_data, self.config.ner_path_config.corrected_streamlit_data_path
            )
            self.logger.info(
                f"Final data saved to {self.config.ner_path_config.corrected_streamlit_data_path}"
            )
            st.success(f"Data saved to {self.config.ner_path_config.corrected_streamlit_data_path}")
            self.session.set("data_hash", self._compute_data_hash(data))
            self.session.set("save_needed", False)

            return True
        except Exception as e:
            self.logger.error(f"Failed to save final data: {str(e)}")
            st.error(f"Error saving final data: {str(e)}")

            return False

class AppUIController:
    """Manages all Streamlit UI components: rendering, filters, and state management."""

    def __init__(self, config: ConfigManager, data_manager: DataManager, data_filter: DataFilter, category_manager: CategoryManager):
        self.config = config
        self.data_manager = data_manager
        self.data_filter = data_filter
        self.category_manager = category_manager
        self.session = SessionState()
    
    def render_sidebar(self):
        """Renders the entire sidebar content."""
        with st.sidebar:
            st.header("Фильтры")
            filter_params, items_per_page = self.render_filter_controls()
            st.markdown("---")
            
            if st.button("💾 Save all changes", use_container_width=True, on_click=self.data_manager.save_final_data):
                pass # The action is in on_click
            
            save_auto = st.checkbox("Autosave", value=True, key="save_auto")
            if save_auto and self.session.get("save_needed", False):
                current_hash = self.data_manager._compute_data_hash(self.session.get("internal_data", []))

                if current_hash != self.session.get("data_hash", ""):
                    with st.spinner("Saving changes..."):
                        if self.data_manager.save_temp_data():
                            st.success("Changes autosaved.")

            st.markdown("---")
            self.render_statistics(self.session.get("internal_data", []))
            st.markdown("### 🐞 Debug: Session State")

            def safe_json(val):
                try:
                    json.dumps(val)
                    return val
                except Exception:
                    return str(val)
                
            safe_state = {k: safe_json(v) for k, v in st.session_state.items()}
            st.json(safe_state, expanded=False)
            
        return filter_params, items_per_page
        
    def render_filter_controls(self) -> Tuple[Tuple[str, bool, bool, bool, bool, bool, bool, str, str], int]:
        """Renders filter controls for category, pagination, and entity search."""
        st.markdown("**Search by entity ID or Text**")
        entity_search_type = st.radio("Search by:", ["text", "id"], index=0, key="entity_search_type_radio", format_func=lambda x: "Text" if x == "text" else "ID", help="Select the search type: by entity text or by entity id.", horizontal=True)
        col_search = st.columns([8, 1, 1])

        if col_search[2].button("✖", key="clear_entity_search_button"):
            self.session.set("entity_search_query", "")
            self.session.set("committed_entity_search_query", "")
            self.session.set("current_page", 1)
            st.rerun()

        entity_search_query = col_search[0].text_input("Enter text or id fo search", value=self.session.get("entity_search_query", ""), key="entity_search_query", help="Enter a word/phrase or id to filter examples.", label_visibility="collapsed")
        
        if col_search[1].button("🔍", key="search_entity_button"):
            self.session.set("committed_entity_search_query", entity_search_query)
            self.session.set("entity_search_type", entity_search_type)
            self.session.set("current_page", 1)

        st.markdown("**Filter by Category**")
        selected_category = st.selectbox("Select category", self.category_manager.get_all() + ["ALL"], index=self.category_manager.get_all().index(st.session_state.get("main_filter_category_selectbox", "ALL")) if st.session_state.get("main_filter_category_selectbox") in self.category_manager.get_all() else len(self.category_manager.get_all()), key="main_filter_category_selectbox", help="Filter examples by entity category.")
        
        items_per_page = st.slider("Examples per page", min_value=1, max_value=self.config.ner_config.items_per_page_max, value=self.config.ner_config.items_per_page_default, key="main_items_per_page_slider", help="Adjust the number of examples per page.")

        show_only_unreviewed = st.checkbox("Show only unreviewed", key="show_only_unreviewed", help="Show examples with unreviewed entities.")
        show_only_reviewed = st.checkbox("Show only reviewed", key="show_only_reviewed", help="Show examples with all entities reviewed.")
        
        st.markdown("**Quick filter for problematic examples:**")
        col1, col2, col3 = st.columns(3)
        with col1: show_overlaps = st.checkbox("Overlaps", key="filter_overlaps", help="Show examples with overlapping entities.")
        with col2: show_empty = st.checkbox("Empty", key="filter_empty", help="Show examples with no entities.")
        with col3: show_invalid = st.checkbox("Invalid indices", key="filter_invalid", help="Show examples with invalid entity indices.")
        
        show_unique_reviewed = st.checkbox("Show only unique reviewed", key="show_unique_reviewed", help="Show only examples with unique reviewed entities (no duplicates based on start, end, and label).")
        
        filter_params = (selected_category, show_only_unreviewed, show_overlaps, show_empty, show_invalid, show_only_reviewed, show_unique_reviewed, self.session.get("committed_entity_search_query", ""), self.session.get("entity_search_type", "text"))
        
        return filter_params, items_per_page
        
    def render_pagination(self, total_items: int, items_per_page: int, position: str = "top") -> int:
        """Renders pagination controls and returns the current page."""
        pages = max(1, (total_items - 1) // items_per_page + 1)
        current_page = self.session.get("current_page", 1)
        current_page = max(1, min(pages, current_page))
        self.session.set("current_page", current_page)

        def goto_page(new_page: int):
            self.session.set("current_page", max(1, min(pages, new_page)))

        col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 1])
        with col1: st.button("←", key=f"prev_page_{position}", use_container_width=True, disabled=current_page == 1, on_click=lambda: goto_page(current_page - 1))
        with col2:
            if current_page > 2:
                st.button(f"{current_page - 2}", key=f"quick_prev2_{position}", use_container_width=True, on_click=lambda: goto_page(current_page - 2))
            else: st.write("")
        with col3:
            if position == "top":
                def on_page_input():
                    val = st.session_state.get("page_number_input", current_page)
                    self.session.set("current_page", max(1, min(pages, val)))
                st.number_input("Page", 1, pages, current_page, key="page_number_input", on_change=on_page_input, label_visibility="collapsed")
                st.write(f"Page {current_page} of {pages}")
            else: st.write(f"Page {current_page} of {pages}")
        with col4:
            if current_page < pages - 1:
                st.button(f"{current_page + 2}", key=f"quick_next2_{position}", use_container_width=True, on_click=lambda: goto_page(current_page + 2))
            else: st.write("")
        with col5: st.button("→", key=f"next_page_{position}", use_container_width=True, disabled=current_page == pages, on_click=lambda: goto_page(current_page + 1))
        
        return current_page

    def render_statistics(self, data: List[Dict]) -> None:
        """Renders statistics for entity categories, each category in its own expander with a column list."""
        stats = {}
        for example in data:
            text = example["text"]
            for entity in example.get("entities", []):
                label = entity["label"]
                if label not in stats:
                    stats[label] = {"count": 0, "entities_names": set()}
                stats[label]["count"] += 1
                entity_text = text[entity["start"] : entity["end"]].strip()
                if entity_text:
                    stats[label]["entities_names"].add(entity_text)
        final_stats = {
            k: {"count": v["count"], "entities_names": sorted(v["entities_names"])}
            for k, v in stats.items()
        }
        st.markdown("**📊 Статистика по категориям**")
        st.json(final_stats)

    def show_annotated_text(self, annotated_fragments: List, default_fill: str) -> None:
        """Renders annotated text with HTML support for overlaps and reviewed status."""
        has_html_inline = any(isinstance(frag, tuple) and frag[1] == "html-inline" for frag in annotated_fragments)
        if has_html_inline:
            html_line = "".join(
                (
                    frag[0]
                    if isinstance(frag, tuple) and frag[1] == "html-inline"
                    else (
                        f'<span style="background-color: {default_fill}; border-radius: 3px; padding: 0 2px; margin: 0 1px; display:inline; vertical-align:baseline; line-height:1.5;">{frag[0]} {"✔️" if frag[3].get("reviewed", False) else ""}</span>'
                        if isinstance(frag, tuple)
                        else frag.replace("<", "&lt;").replace(">", "&gt;")
                    )
                )
                for frag in annotated_fragments
            )
            st.markdown(f'<div style="display:inline; word-break:break-word;">{html_line}</div>', unsafe_allow_html=True)
        else:
            modified_fragments = [
                ((f"{frag[0]} ✔️", frag[1], frag[2]) if isinstance(frag, tuple) and frag[3].get("reviewed", False) else frag)
                for frag in annotated_fragments
            ]
            annotated_text(*modified_fragments)

    def render_example(self, example: Dict, original_index: int, page_idx: int) -> None:
        """Renders a single example with entity highlighting and editing controls."""
        st.markdown("---")
        st.markdown(f"### Example {original_index + 1} (ID: {example['id']})")
        full_text = example["text"]
        entities = example.get("entities", [])
        sorted_entities = sorted(enumerate(entities), key=lambda x: x[1]["start"])
        entity_spans = []
        for ent_idx, (orig_idx, entity) in enumerate(sorted_entities):
            start = self.session.get(f"start_{original_index}_{orig_idx}", entity["start"])
            end = self.session.get(f"end_{original_index}_{orig_idx}", entity["end"])
            label = self.session.get(f"label_{original_index}_{orig_idx}", entity["label"])
            if 0 <= start < end <= len(full_text):
                entity_spans.append({"start": start, "end": end, "label": label, "ent_idx": ent_idx, "reviewed": entity.get("reviewed", False)})
            else:
                st.warning(f"Invalid indices for entity '{entity['label']}' in example {example['id']}")
        
        entity_spans.sort(key=lambda x: x["start"])
        merged = []
        for span in entity_spans:
            if not merged or span["start"] >= merged[-1]["end"]:
                merged.append({"start": span["start"], "end": span["end"], "labels": [span["label"]], "ent_idxs": [span["ent_idx"]], "reviewed": [span["reviewed"]]})
            else:
                last = merged[-1]
                last["end"] = max(last["end"], span["end"])
                last["labels"].append(span["label"])
                last["ent_idxs"].append(span["ent_idx"])
                last["reviewed"].append(span["reviewed"])

        annotated_fragments = []
        last_idx = 0
        for group in merged:
            if last_idx < group["start"]:
                annotated_fragments.append(full_text[last_idx : group["start"]])
            frag = full_text[group["start"] : group["end"]]
            main_label = group["labels"][0]
            reviewed = all(group["reviewed"])
            fill = self.config.ner_config.entity_colors.get(main_label, "#fff")
            if len(group["labels"]) > 1:
                html = f'<span style="background-color: {fill}; border: 2px solid #d9534f; border-radius: 3px; padding: 0 2px; margin: 0 1px; display:inline; vertical-align:baseline; line-height:1.5;">{frag} {"✔️" if reviewed else ""}</span>'
                annotated_fragments.append((html, "html-inline"))
            else:
                annotated_fragments.append((frag, main_label, fill, {"reviewed": reviewed}))
            last_idx = group["end"]
        
        if last_idx < len(full_text):
            annotated_fragments.append(full_text[last_idx:])
        default_fill = self.config.ner_config.entity_colors.get(entities[0]["label"], "#fff") if entities else "#fff"
        self.show_annotated_text(annotated_fragments, default_fill)

        if entities:
            overlaps_for_editor, _ = self._get_overlaps(entities)
            for ent_idx, entity in enumerate(entities):
                with st.container(key=f"entity_editor_container_{original_index}_{ent_idx}"):
                    self._render_entity_editor(ent_idx, entity, original_index, overlaps_for_editor)

        self._render_new_entity_form(original_index, full_text, page_idx)

    def _get_overlaps(self, entities: List[Dict]) -> Tuple[Set[int], Set[int]]:
        """Identifies overlapping and duplicate entities."""
        overlaps = set()
        duplicates = set()
        seen = {}
        sorted_entities_with_indices = sorted(enumerate(entities), key=lambda x: x[1]["start"])
        for i, (orig_i, e1) in enumerate(sorted_entities_with_indices):
            key = (e1["start"], e1["end"], e1["label"])
            if key in seen:
                duplicates.add(orig_i)
                duplicates.update(seen[key])
            else:
                seen[key] = {orig_i}

            for j in range(i + 1, len(sorted_entities_with_indices)):
                orig_j, e2 = sorted_entities_with_indices[j]
                if e1["end"] <= e2["start"]: break
                if not (e1["end"] <= e2["start"] or e2["end"] <= e1["start"]):
                    overlaps.add(orig_i)
                    overlaps.add(orig_j)
        return overlaps, duplicates

    def _render_entity_editor(self, ent_idx: int, entity: Dict, original_index: int, overlaps: Set[int]) -> None:
        """Renders editor for a single entity."""
        full_text = self.session.get("internal_data")[original_index]["text"]
        key_prefix = f"entity_{original_index}_{ent_idx}"
        start = self.session.get(f"entity_start_{key_prefix}", entity["start"])
        end = self.session.get(f"entity_end_{key_prefix}", entity["end"])
        
        entity_text = full_text[start:end] if 0 <= start < end <= len(full_text) else "Invalid indices"
        if entity_text == "Invalid indices": st.warning("⚠️ Entity range is out of text bounds!")

        with st.container(border=True):
            if ent_idx in overlaps: st.warning("⚠️ This entity overlaps with another. Check ranges!")
            st.markdown(f'<span style="text-decoration: underline;">{entity_text.replace(" ", "▮")}</span> {"✔️" if entity.get("reviewed", False) else ""}', unsafe_allow_html=True)
            
            st.slider("Range (start, end)", 0, len(full_text), (start, end), key=f"entity_range_slider_{key_prefix}", on_change=lambda: self._handle_range_change(original_index, ent_idx, key_prefix, entity))
            
            col_label, col_info, col_mark, col_review, col_delete = st.columns([3, 1, 3, 1, 1], vertical_alignment="bottom")
            cat_options = self.category_manager.get_all()
            
            with col_label:
                selected_label = st.selectbox("Label", cat_options + ["+ New category..."], index=(cat_options.index(entity["label"]) if entity["label"] in cat_options else 0), key=f"entity_label_select_{key_prefix}", on_change=lambda: self._handle_label_change(original_index, ent_idx, key_prefix, entity))
                
                if selected_label == "+ New category...":
                    new_cat = st.text_input("Enter new category", key=f"custom_cat_{key_prefix}")

                    if new_cat and self.category_manager.add_custom(new_cat):
                        self.session.set(f"entity_label_select_{key_prefix}", new_cat)
                        st.rerun()

            with col_info:
                with st.popover(label="", icon="ℹ️", help="Category information"):
                    st.markdown("**Category Information**")
                    table_md = "| Category | Description |\n|---|---|\n"

                    for cat, desc in self.config.ner_config.category_info.items():
                        table_md += f"| `{cat}` | {desc} |\n"
                    st.markdown(table_md)
            
            with col_review: st.button("👁️", key=f"entity_reviewed_btn_{key_prefix}", help="Mark as reviewed", on_click=lambda: self._mark_entity_reviewed(original_index, ent_idx, entity), disabled=entity.get("reviewed", False), use_container_width=True)
            with col_mark: st.button("Find and mark similar", icon="🔎", key=f"entity_mark_all_btn_{key_prefix}", on_click=lambda: self._mark_all_similar(original_index, entity), use_container_width=True)
            with col_delete: 
                if st.button("", icon="🗑️", help="Delete entity", key=f"entity_delete_btn_{key_prefix}", use_container_width=True):
                    self._delete_entity(original_index, ent_idx)
                    st.rerun()

    def _handle_label_change(self, original_index, ent_idx, key_prefix, entity):
        new_label = st.session_state[f"entity_label_select_{key_prefix}"]

        if entity.get("reviewed", False):
            if st.button("Confirm Change", key=f"confirm_label_{key_prefix}"):
                self._update_entity(original_index, ent_idx, key_prefix, new_label=new_label)
                self.session.set("save_needed", True)
                st.rerun()
        else:
            self._update_entity(original_index, ent_idx, key_prefix, new_label=new_label)
            self.session.set("save_needed", True)

    def _handle_range_change(self, original_index, ent_idx, key_prefix, entity):
        start, end = st.session_state[f"entity_range_slider_{key_prefix}"]

        if entity.get("reviewed", False):
            if st.button("Confirm Change", key=f"confirm_range_{key_prefix}"):
                self._update_entity(original_index, ent_idx, key_prefix, start=start, end=end)
                self.session.set("save_needed", True)
                st.rerun()
        else:
            self._update_entity(original_index, ent_idx, key_prefix, start=start, end=end)
            self.session.set("save_needed", True)

    def _mark_entity_reviewed(self, original_index, ent_idx, entity):
        entities = self.session.get("internal_data")[original_index]["entities"]

        if 0 <= ent_idx < len(entities):
            entities[ent_idx]["reviewed"] = True
            entities[ent_idx]["last_modified"] = datetime.now().isoformat()
            self.data_manager.save_temp_data()

    def _mark_all_similar(self, original_index, entity):
        full_text = self.session.get("internal_data")[original_index]["text"]
        start, end = entity["start"], entity["end"]
        word = full_text[start:end]

        if not word.strip():
            st.toast("Select a non-empty word.", icon="⚠️")
            return
        
        new_label = entity["label"]
        data = self.session.get("internal_data")
        count_new, count_updated = 0, 0
        
        for ex_idx, ex in enumerate(data):
            text = ex["text"]

            for match in re.finditer(re.escape(word), text, re.IGNORECASE):
                start_m, end_m = match.span()
                is_new = True

                for e in ex.get("entities", []):
                    if e["start"] == start_m and e["end"] == end_m:
                        if e["label"] != new_label:
                            e["label"] = new_label
                            count_updated += 1

                        if not e.get("reviewed", False):
                            e["reviewed"] = True
                            e["last_modified"] = datetime.now().isoformat()
                            count_updated += 1
                        is_new = False
                        break

                if is_new:
                    ex["entities"].append({"start": start_m, "end": end_m, "label": new_label, "reviewed": True, "last_modified": datetime.now().isoformat()})
                    count_new += 1

        self.data_manager.save_temp_data()
        st.toast(f"Added {count_new}, updated {count_updated} for '{word}'.", icon="✅")

    def _delete_entity(self, original_index: int, ent_idx: int) -> None:
        """Deletes an entity from the dataset."""
        entities = self.session.get("internal_data")[original_index]["entities"]

        if 0 <= ent_idx < len(entities):
            entities.pop(ent_idx)
            self.data_manager.save_temp_data()

    def _update_entity(self, original_index: int, ent_idx: int, key_prefix: str, new_label: str = None, start: int = None, end: int = None) -> None:
        """Updates entity data from session state."""
        entities = self.session.get("internal_data")[original_index]["entities"]

        if 0 <= ent_idx < len(entities):
            entity = entities[ent_idx]
            entity["start"] = start if start is not None else entity["start"]
            entity["end"] = end if end is not None else entity["end"]
            entity["label"] = new_label if new_label is not None else entity["label"]
            entity["reviewed"] = False
            entity["last_modified"] = datetime.now().isoformat()

            self.data_manager.save_temp_data()
            st.rerun()

    def _render_new_entity_form(self, original_index: int, full_text: str, page_idx: int) -> None:
        """Renders form for adding new entities by word selection."""
        with st.expander("Add new entity by word", icon="➕"):
            word_key = f"word_add_{original_index}_{page_idx}"
            multiselect_key = f"multiselect_words_{original_index}_{page_idx}"
            label_key = f"label_add_{original_index}_{page_idx}"
            clear_inputs_key = f"clear_inputs_{original_index}_{page_idx}"

            if self.session.get(clear_inputs_key, False):
                self.session.set(word_key, "")
                self.session.set(multiselect_key, [])
                self.session.set(clear_inputs_key, False)

            word = st.text_input("Word (manual input)", value=self.session.get(word_key, ""), key=word_key)
            unique_words = sorted(list(set(re.findall(r"\b\w+\b", full_text.lower()))))
            selected_words = st.multiselect("Select words from text", options=unique_words, default=self.session.get(multiselect_key, []), key=multiselect_key)
            cat_options = self.category_manager.get_all()
            label = st.selectbox("Category", cat_options + ["+ New category..."], key=label_key)

            if label == "+ New category...":
                new_cat = st.text_input("Enter new category", key=f"custom_cat_add_{original_index}_{page_idx}")

                if new_cat and self.category_manager.add_custom(new_cat):
                    self.session.set(label_key, new_cat)
                    self.session.set("save_needed", True)
                    st.rerun()

            def add_entities(words_to_add: List[str]):
                if not words_to_add:
                    st.warning("Enter a word or select words from the list.")
                    return
                
                entities = self.session.get("internal_data")[original_index]["entities"]
                total_new = 0

                for word_to_add in words_to_add:
                    for match in re.finditer(re.escape(word_to_add), full_text, re.IGNORECASE):
                        start_idx, end_idx = match.span()

                        if not any(e["start"] == start_idx and e["end"] == end_idx for e in entities):
                            entities.append({"start": start_idx, "end": end_idx, "label": label, "reviewed": False, "last_modified": datetime.now().isoformat()})
                            total_new += 1

                if total_new:
                    self.data_manager.save_temp_data()
                    st.info(f"Added {total_new} new entities.")
                    self.session.set(clear_inputs_key, True)
                    st.rerun()
                else:
                    st.warning("All selected words are already annotated.")

            if st.button("Add", key=f"btn_add_{original_index}_{page_idx}", use_container_width=True):
                words_to_add = selected_words if selected_words else ([word] if word else [])
                add_entities(words_to_add)


class NERDatasetReviewer:
    """Main application class for the NER Dataset Reviewer."""

    def __init__(self, config: ConfigManager = None):
        self.config = config or ConfigManager()
        self.logger = AppLogger("ner_reviewer").get_logger
        self.file_utils = FileUtils(self.logger)
        self.session = SessionState()
        self.data_manager = DataManager(self.config, self.file_utils, self.logger)
        self.data_filter = DataFilter(self.logger)
        self.category_manager = CategoryManager(self.config, self.logger)
        self.ui_controller = AppUIController(
            self.config, self.data_manager, self.data_filter, self.category_manager
        )

    def _get_original_index(self, example: Dict) -> Optional[int]:
        """Finds the original index of an example in internal_data."""
        data = self.session.get("internal_data", [])
        
        for idx, ex in enumerate(data):
            if ex is example:
                return idx
            
        return None

    def run(self) -> None:
        """Runs the Streamlit application with sidebar and optimized layout."""
        st.set_page_config(page_title="NER Dataset Reviewer", layout="wide")
        st.markdown("""<style>.block-container {max-width: 1000px !important; padding-left: 2rem; padding-right: 2rem;}</style>""", unsafe_allow_html=True)
        st.title("🔧 Manual NER Dataset Correction")

        if self.session.get("internal_data") is None:
            self.data_manager.load_data()

        data = self.session.get("internal_data", [])
        filter_params, items_per_page = self.ui_controller.render_sidebar()
        filtered_data = self.data_filter.filter(data, *filter_params)
        self.data_filter.get_filter_feedback(filtered_data, *filter_params)

        if not filtered_data:
            st.info("No examples to display.")
            return

        page = self.ui_controller.render_pagination(len(filtered_data), items_per_page, position="top")
        start = (page - 1) * items_per_page
        end = start + items_per_page

        for idx_in_page, example in enumerate(filtered_data[start:end]):
            original_index = self._get_original_index(example)
            if original_index is not None:
                self.ui_controller.render_example(example, original_index, idx_in_page)
            else:
                st.warning(f"Could not find original index for example ID: {example['id']}")
                
        self.ui_controller.render_pagination(len(filtered_data), items_per_page, position="bottom")

def main():
    """Entry point for the application."""
    reviewer = NERDatasetReviewer()
    reviewer.run()

if __name__ == "__main__":
    main()
"""Notion document loader for fetching database pages via Notion API."""

from typing import Optional
from langchain_core.documents import Document
from notion_client import Client

from src.config import NOTION_TOKEN, NOTION_DATABASE_ID


def get_notion_client() -> Client:
    """Get an authenticated Notion client."""
    if not NOTION_TOKEN:
        raise ValueError("NOTION_TOKEN environment variable is not set")
    return Client(auth=NOTION_TOKEN)


def extract_text_from_rich_text(rich_text: list) -> str:
    """Extract plain text from Notion rich text array."""
    return "".join(item.get("plain_text", "") for item in rich_text)


def extract_property_value(prop: dict) -> str:
    """Extract text value from a Notion page property."""
    prop_type = prop.get("type", "")
    
    if prop_type == "title":
        return extract_text_from_rich_text(prop.get("title", []))
    elif prop_type == "rich_text":
        return extract_text_from_rich_text(prop.get("rich_text", []))
    elif prop_type == "number":
        num = prop.get("number")
        return str(num) if num is not None else ""
    elif prop_type == "select":
        select = prop.get("select")
        return select.get("name", "") if select else ""
    elif prop_type == "multi_select":
        options = prop.get("multi_select", [])
        return ", ".join(opt.get("name", "") for opt in options)
    elif prop_type == "date":
        date = prop.get("date")
        if date:
            start = date.get("start", "")
            end = date.get("end", "")
            return f"{start} - {end}" if end else start
        return ""
    elif prop_type == "checkbox":
        return "Yes" if prop.get("checkbox") else "No"
    elif prop_type == "url":
        return prop.get("url", "") or ""
    elif prop_type == "email":
        return prop.get("email", "") or ""
    elif prop_type == "phone_number":
        return prop.get("phone_number", "") or ""
    elif prop_type == "people":
        people = prop.get("people", [])
        return ", ".join(p.get("name", "") for p in people if p.get("name"))
    elif prop_type == "status":
        status = prop.get("status")
        return status.get("name", "") if status else ""
    elif prop_type == "relation":
        # Relations are IDs, not useful as text
        return ""
    elif prop_type == "formula":
        formula = prop.get("formula", {})
        f_type = formula.get("type", "")
        return str(formula.get(f_type, ""))
    elif prop_type == "rollup":
        rollup = prop.get("rollup", {})
        r_type = rollup.get("type", "")
        if r_type == "array":
            # Recursively extract from array items
            items = rollup.get("array", [])
            return ", ".join(extract_property_value(item) for item in items)
        return str(rollup.get(r_type, ""))
    
    return ""


def extract_text_from_block(block: dict) -> str:
    """Extract text content from a Notion block."""
    block_type = block.get("type", "")
    block_data = block.get(block_type, {})
    
    # Text-based blocks
    if block_type in ("paragraph", "heading_1", "heading_2", "heading_3", 
                      "bulleted_list_item", "numbered_list_item", "quote", "callout"):
        rich_text = block_data.get("rich_text", [])
        text = extract_text_from_rich_text(rich_text)
        
        if block_type == "heading_1":
            return f"# {text}"
        elif block_type == "heading_2":
            return f"## {text}"
        elif block_type == "heading_3":
            return f"### {text}"
        elif block_type in ("bulleted_list_item", "numbered_list_item"):
            return f"• {text}"
        elif block_type == "quote":
            return f"> {text}"
        return text
    
    elif block_type == "code":
        rich_text = block_data.get("rich_text", [])
        code = extract_text_from_rich_text(rich_text)
        language = block_data.get("language", "")
        return f"```{language}\n{code}\n```"
    
    elif block_type == "toggle":
        rich_text = block_data.get("rich_text", [])
        return extract_text_from_rich_text(rich_text)
    
    elif block_type == "to_do":
        rich_text = block_data.get("rich_text", [])
        checked = block_data.get("checked", False)
        checkbox = "[x]" if checked else "[ ]"
        return f"{checkbox} {extract_text_from_rich_text(rich_text)}"
    
    elif block_type == "divider":
        return "---"
    
    elif block_type in ("table_of_contents", "breadcrumb", "child_page", "child_database"):
        return ""
    
    return ""


def get_page_url(page_id: str) -> str:
    """Construct a Notion page URL from page ID."""
    clean_id = page_id.replace("-", "")
    return f"https://notion.so/{clean_id}"


def fetch_page_blocks(client: Client, page_id: str) -> list[dict]:
    """Fetch all blocks from a page, handling pagination."""
    blocks = []
    cursor = None
    
    while True:
        response = client.blocks.children.list(
            block_id=page_id,
            start_cursor=cursor
        )
        blocks.extend(response.get("results", []))
        
        if not response.get("has_more"):
            break
        cursor = response.get("next_cursor")
    
    return blocks


def fetch_page_content(client: Client, page_id: str, depth: int = 0, max_depth: int = 5) -> str:
    """Fetch all content from a page's blocks, recursively handling nested blocks."""
    if depth > max_depth:
        return ""
    
    blocks = fetch_page_blocks(client, page_id)
    content_parts = []
    
    for block in blocks:
        text = extract_text_from_block(block)
        if text:
            content_parts.append(text)
        
        # Handle nested blocks (e.g., inside toggles)
        if block.get("has_children") and block.get("type") not in ("child_page", "child_database"):
            nested = fetch_page_content(client, block["id"], depth + 1, max_depth)
            if nested:
                content_parts.append(nested)
    
    return "\n\n".join(content_parts)


def get_data_source_ids(client: Client, database_id: str) -> list[str]:
    """Retrieve database and extract data source IDs."""
    database = client.databases.retrieve(database_id=database_id)
    data_sources = database.get("data_sources", [])
    return [ds.get("id") for ds in data_sources if ds.get("id")]


def query_data_source(client: Client, data_source_id: str) -> list[dict]:
    """Query all pages from a Notion data source, handling pagination."""
    pages = []
    cursor = None
    
    while True:
        # Use POST request to query data source
        response = client.request(
            path=f"data_sources/{data_source_id}/query",
            method="POST",
            body={"start_cursor": cursor} if cursor else {}
        )
        pages.extend(response.get("results", []))
        
        if not response.get("has_more"):
            break
        cursor = response.get("next_cursor")
    
    return pages


def query_database(client: Client, database_id: str) -> list[dict]:
    """Query all pages from a Notion database via its data sources."""
    # Get data source IDs from database
    data_source_ids = get_data_source_ids(client, database_id)
    
    if not data_source_ids:
        print("Warning: No data sources found in database")
        return []
    
    # Query each data source and collect pages
    all_pages = []
    for ds_id in data_source_ids:
        pages = query_data_source(client, ds_id)
        all_pages.extend(pages)
    
    return all_pages


def load_database_page(client: Client, page: dict, include_content: bool = True) -> Optional[Document]:
    """
    Load a single database page as a Document.
    
    Extracts both properties (columns) and page content (body).
    """
    page_id = page.get("id", "")
    properties = page.get("properties", {})
    
    # Extract title from properties
    title = "Untitled"
    for prop_name, prop_value in properties.items():
        if prop_value.get("type") == "title":
            title = extract_property_value(prop_value) or "Untitled"
            break
    
    # Build property text
    prop_lines = [f"# {title}", ""]
    for prop_name, prop_value in properties.items():
        if prop_value.get("type") == "title":
            continue  # Already used as heading
        value = extract_property_value(prop_value)
        if value:
            prop_lines.append(f"**{prop_name}**: {value}")
    
    # Get page body content
    body_content = ""
    if include_content:
        try:
            body_content = fetch_page_content(client, page_id)
        except Exception as e:
            print(f"Warning: Failed to fetch content for page {title}: {e}")
    
    # Combine properties and content
    content_parts = prop_lines
    if body_content:
        content_parts.append("")
        content_parts.append("---")
        content_parts.append("")
        content_parts.append(body_content)
    
    full_content = "\n".join(content_parts)
    
    if not full_content.strip():
        return None
    
    return Document(
        page_content=full_content,
        metadata={
            "source": f"notion:{title}",
            "notion_page_id": page_id,
            "notion_url": get_page_url(page_id),
            "title": title.lower(),  # lowercase for consistent filtering
            "file_type": "notion"
        }
    )


def load_notion_database(
    database_id: Optional[str] = None,
    include_content: bool = True
) -> list[Document]:
    """
    Load all pages from a Notion database.
    
    Args:
        database_id: The Notion database ID. Defaults to NOTION_DATABASE_ID.
        include_content: Whether to fetch page body content (slower but more complete).
        
    Returns:
        List of Document objects, one per database page.
    """
    database_id = database_id or NOTION_DATABASE_ID
    
    if not database_id:
        raise ValueError("No database ID provided and NOTION_DATABASE_ID is not set")
    
    client = get_notion_client()
    pages = query_database(client, database_id)
    
    documents = []
    for i, page in enumerate(pages, 1):
        print(f"  Loading Notion page {i}/{len(pages)}...")
        try:
            doc = load_database_page(client, page, include_content=include_content)
            if doc:
                documents.append(doc)
        except Exception as e:
            page_id = page.get("id", "unknown")
            print(f"Warning: Failed to load page {page_id}: {e}")
    
    return documents


def load_notion_documents() -> list[Document]:
    """
    Load all documents from the configured Notion database.
    
    This is the main entry point for Notion document loading.
    """
    if not NOTION_TOKEN or not NOTION_DATABASE_ID:
        return []
    
    return load_notion_database(include_content=True)

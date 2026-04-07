"""CVD Neo4j Knowledge Graph Query Tool for DeerFlow.

Query the CVD equipment knowledge graph stored in Neo4j.
Supports natural language questions about equipment anomalies, root causes, and actions.
"""

import os
from typing import Annotated, Literal

from langchain_core.runnables import RunnableConfig
from langchain.tools import InjectedToolCallId, ToolRuntime, tool

from deerflow.agents.thread_state import ThreadState

# Neo4j connection config from environment
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:17687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")


def _query_neo4j(cypher: str) -> list[dict]:
    """Execute a Cypher query against Neo4j and return results."""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        results = session.run(cypher)
        return [dict(record) for record in results]


def _build_cypher_from_question(question: str) -> tuple[str, dict]:
    """Build a Cypher query from a natural language CVD question.

    Returns:
        Tuple of (cypher_query, parameters_dict)
    """
    question_lower = question.lower()

    # Pattern: equipment + anomaly
    if "异常" in question or "报警" in question or "故障" in question or "alarm" in question_lower or "fault" in question_lower:
        # Try to extract equipment name
        cypher = """
        MATCH (e:Equipment)-[:HAS_ALARM]->(a:AlarmCode)-[:CAUSED_BY]->(r:RootCause)
        WHERE toLower(e.name) CONTAINS toLower($keyword)
           OR toLower($keyword) CONTAINS toLower(e.name)
        RETURN e.name AS equipment, a.code AS alarm_code, a.description AS alarm_desc,
               r.name AS root_cause, r.action AS recommended_action
        LIMIT 10
        """
        # Extract keyword (first notable noun)
        keyword = question.replace("异常", "").replace("报警", "").replace("故障", "").strip()
        keyword = keyword.split("原因")[0].split("解决")[0].strip()
        return cypher, {"keyword": keyword}

    # Pattern: root cause → action
    if "原因" in question or "为什么" in question or "root cause" in question_lower:
        cypher = """
        MATCH (r:RootCause)-[:REQUIRES]->(a:Action)
        WHERE toLower($keyword) CONTAINS toLower(r.name)
        RETURN r.name AS root_cause, a.name AS action, a.description AS action_desc
        LIMIT 10
        """
        keyword = question.split("原因")[0].split("为什么")[0].strip()
        return cypher, {"keyword": keyword}

    # Pattern: action / SOP
    if "处理" in question or "步骤" in question or "怎么做" in question or "action" in question_lower or "sop" in question_lower:
        cypher = """
        MATCH (r:RootCause)-[:REQUIRES]->(a:Action)
        RETURN r.name AS root_cause, a.name AS action, a.description AS action_desc
        LIMIT 15
        """
        return cypher, {"keyword": question}

    # Default: get all DefectSOPRule
    cypher = """
    MATCH (d:DefectSOPRule)
    RETURN d.defect AS defect, d.cause AS cause, d.action AS action
    LIMIT 20
    """
    return cypher, {"keyword": question}


@tool("neo4j_query", parse_docstring=True)
def neo4j_query_tool(
    runtime: ToolRuntime[Literal, ThreadState],
    question: Annotated[str, InjectedToolCallId],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    """Query the CVD equipment knowledge graph from Neo4j.

    Use this tool when the user asks about:
    - Equipment anomalies and alarms (e.g., "RF_VDC 异常怎么处理")
    - Root cause analysis (e.g., "为什么会报警")
    - Recommended actions / SOP steps (e.g., "怎么处理")
    - Relationships between equipment, alarms, and actions

    Args:
        question: Natural language question about CVD equipment.
            Examples:
            - "RF_VDC 报警的原因是什么"
            - "Chamber pressure high 的处理步骤"
            - "设备异常应该如何处理"
    """
    try:
        cypher, params = _build_cypher_from_question(question)
        results = _query_neo4j(cypher.format(**params))

        if not results:
            return f"No results found in Neo4j for: {question}"

        output = f"Found {len(results)} results:\n\n"
        for i, r in enumerate(results, 1):
            output += f"{i}. {dict(r)}\n"

        return output

    except Exception as e:
        return f"Neo4j query error: {str(e)}"

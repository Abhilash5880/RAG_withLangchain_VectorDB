# Install dependencies
# uv add langchain langchain-google-genai langchain-chroma rank-bm25
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

documents = [
    Document(
        page_content=(
            "The AcmeNet AX-7742X is a dual-band Wi-Fi 6 router for busy homes and "
            "small offices. It has four gigabit Ethernet ports, supports WPA3, and "
            "uses Quality of Service (QoS) to prioritize video calls and gaming traffic."
        ),
        metadata={
            "type": "product",
            "record_id": "PROD-7742X",
            "sku": "AX-7742X",
            "source": "product_catalog",
        },
    ),
    Document(
        page_content=(
            "If Wi-Fi is connected but websites will not load, restart the modem and "
            "router, then check whether the internet or WAN status light is red. If "
            "only one device is affected, forget the wireless network on that device "
            "and reconnect using the current password."
        ),
        metadata={
            "type": "troubleshooting",
            "record_id": "HELP-NET-104",
            "category": "connectivity",
            "source": "support_guide",
        },
    ),
    Document(
        page_content=(
            "Error E_CONN_REFUSED (connection refused) means the remote host was "
            "reachable but no service accepted the TCP connection on the requested "
            "port. Verify the service is running, confirm the host and port, and "
            "inspect firewall or security-group rules."
        ),
        metadata={
            "type": "error",
            "record_id": "ERR-E_CONN_REFUSED",
            "error_code": "E_CONN_REFUSED",
            "source": "error_reference",
        },
    ),
    Document(
        page_content=(
            "When an account reports invalid credentials after a password reset, "
            "sign out of all active sessions and authenticate again with the new "
            "password. If multi-factor authentication is enabled, use the newest "
            "one-time code; backup codes are single-use."
        ),
        metadata={
            "type": "authentication",
            "record_id": "HELP-AUTH-022",
            "category": "account_access",
            "source": "support_guide",
        },
    ),
    Document(
        page_content=(
            "API error HTTP 429 means the client exceeded its rate limit. Respect "
            "the Retry-After response header, use exponential backoff with jitter, "
            "and avoid retrying every request immediately. The limit is scoped to "
            "the API key and endpoint."
        ),
        metadata={
            "type": "api_error",
            "record_id": "ERR-HTTP-429",
            "error_code": "HTTP_429",
            "source": "api_runbook",
        },
    ),
    Document(
        page_content=(
            "The AX-7742X router may lose its WAN connection after firmware "
            "3.2.1 is installed. Save the configuration, install firmware 3.2.2 "
            "or later, then reboot the modem before restarting the router. This "
            "issue does not affect the AX-8810 model."
        ),
        metadata={
            "type": "known_issue",
            "record_id": "ISSUE-FW-321",
            "sku": "AX-7742X",
            "firmware_version": "3.2.1",
            "source": "release_notes",
        },
    ),
    Document(
        page_content=(
            "For slow video calls or online games, connect the computer using "
            "Ethernet where possible, pause large downloads, and enable QoS on "
            "the router. QoS can prioritize latency-sensitive voice and video "
            "traffic, but it cannot increase the internet speed provided by "
            "your ISP."
        ),
        metadata={
            "type": "troubleshooting",
            "record_id": "HELP-PERF-018",
            "category": "performance",
            "source": "support_guide",
        },
    ),
    Document(
        page_content=(
            "DNS error E_DNS_004 occurs when a device cannot resolve a domain "
            "name to an IP address. Check the configured DNS server, try a "
            "different resolver, flush the local DNS cache, and compare the "
            "result with a direct connection to the service's IP address."
        ),
        metadata={
            "type": "error",
            "record_id": "ERR-E_DNS_004",
            "error_code": "E_DNS_004",
            "source": "error_reference",
        },
    ),
    Document(
        page_content=(
            "A customer may return an unopened router within 30 days of delivery "
            "for a refund. Include the order number and original accessories. "
            "Products damaged by liquid or modified firmware are not eligible "
            "for the standard return policy."
        ),
        metadata={
            "type": "policy",
            "record_id": "POL-RET-030",
            "category": "returns",
            "source": "store_policy",
        },
    ),
]

print(f"Loaded {len(documents)} documents for hybrid search demo.")

vectorstore = Chroma.from_documents(
    documents,
    embeddings_model,
    collection_name="hybrid_search",
)

# create vector retriever
vector_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3},
)
print("Vector retriever created.")

# create BM25 retriever
bm25_retriever = BM25Retriever.from_documents(
    documents,
    k=3,  # return top 3
)
print("BM25 retriever created.")


def hybrid_search(vector_results, bm25_results, k=3):
    """Combine already-fetched rankings using reciprocal rank fusion."""
    documents_by_id = {}
    scores = {}

    # A document ranked highly by either method gets a stronger fused score.
    for results, weight in ((vector_results, 0.5), (bm25_results, 0.5)):
        for rank, document in enumerate(results, start=1):
            # Use record_id to recognize the same document in both result lists.
            document_id = document.metadata.get("record_id", document.page_content)
            documents_by_id[document_id] = document
            scores[document_id] = scores.get(document_id, 0.0) + weight / (60 + rank)

    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    return [documents_by_id[document_id] for document_id in ranked_ids[:k]]


print("Hybrid search ready.")


def test_query(query, name, results):
    """Print a preview of one retriever's results for the query."""
    print(f"\n{name} - Query: {query!r}")
    for index, document in enumerate(results[:3], start=1):
        preview = document.page_content[:180].replace("\n", " ")
        print(f"{index}. [{document.metadata.get('type', 'unknown')}] {preview}...")
    return results


test_queries = [
    "AX-7742X specifications and supported wireless security",
    "E_CONN_REFUSED error",
    "Why did the router lose internet after a firmware update?",
    "I changed my password and can't sign in with my authenticator code",
    "How should a client handle API throttling?",
    "Improve lag during video calls and online gaming",
    "E_DNS_004",
    "Can I get a refund for a router with modified firmware?",
]

# Print each retriever's results separately, then show how rank fusion combines them.
for query in test_queries:
    # Fetch each ranking once so the hybrid result uses exactly these results.
    vector_results = vector_retriever.invoke(query)
    bm25_results = bm25_retriever.invoke(query)
    hybrid_results = hybrid_search(vector_results, bm25_results)

    test_query(query, "Vector", vector_results)
    test_query(query, "BM25", bm25_results)
    test_query(query, "Hybrid", hybrid_results)
"""
Regression test for the Collections add-paper persistence bug found during
the Phase 2 browser audit: POST /api/v1/collections/{id}/papers/{paper_id}
returned {"success": true} / 200 while the underlying INSERT into
collection_papers silently failed (NotNullViolationError on added_at,
which had no DB-level default despite the ORM declaring
server_default=func.now() -- see alembic/versions/collection_papers_added_at_default.py).

Exercises the real router (api/routers/collections.py) against the real
database configured via DATABASE_URL/.env -- the bug was a DB-schema/
transaction-timing issue that an in-memory or mocked session would not
have caught. The collection created here is deleted at the end (cascade-
deletes its collection_papers row), and no existing paper row is created,
modified, or deleted, so this does not touch the 122-paper corpus.
"""
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from api.main import app
from api.core.database import AsyncSessionLocal
from api.models.tables import Paper


@pytest.mark.asyncio
async def test_add_paper_to_collection_persists_and_remove_works():
    async with AsyncSessionLocal() as session:
        existing_paper = (await session.execute(select(Paper.id).limit(1))).scalar_one_or_none()
    assert existing_paper is not None, "no papers in corpus to test against"
    paper_id = str(existing_paper)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        collection_id = None
        try:
            create_resp = await client.post(
                "/api/v1/collections",
                json={"name": "pytest-regression-add-paper-tmp"},
            )
            assert create_resp.status_code == 200
            collection_id = create_resp.json()["id"]

            # Add the paper -- this is the call that used to return a
            # false-positive {"success": true}.
            add_resp = await client.post(f"/api/v1/collections/{collection_id}/papers/{paper_id}")
            assert add_resp.status_code == 200
            assert add_resp.json() == {"success": True}

            # The real regression check: did it actually persist?
            papers_resp = await client.get(f"/api/v1/collections/{collection_id}/papers")
            assert papers_resp.status_code == 200
            paper_ids = [p["id"] for p in papers_resp.json()]
            assert paper_id in paper_ids, (
                "paper missing from collection after add -- persistence bug regressed"
            )

            # Adding the same paper again should stay idempotent, not error.
            add_again_resp = await client.post(f"/api/v1/collections/{collection_id}/papers/{paper_id}")
            assert add_again_resp.status_code == 200

            # Removal should work and actually take effect.
            remove_resp = await client.delete(f"/api/v1/collections/{collection_id}/papers/{paper_id}")
            assert remove_resp.status_code == 200
            assert remove_resp.json() == {"success": True}

            papers_after_remove = await client.get(f"/api/v1/collections/{collection_id}/papers")
            assert papers_after_remove.status_code == 200
            assert paper_id not in [p["id"] for p in papers_after_remove.json()]
        finally:
            if collection_id is not None:
                await client.delete(f"/api/v1/collections/{collection_id}")

from fastapi import APIRouter
from api.schemas.schemas import FlowchartRequest, FlowchartResponse

router = APIRouter(prefix="/api/v1", tags=["Flowchart"])


@router.post("/flowchart", response_model=FlowchartResponse)
async def generate_flowchart(request: FlowchartRequest):
    """Generate a Mermaid.js flowchart from a question + explanation."""
    from src.ui.flowchart import generate_flowchart as gen_fc

    diagram = gen_fc(request.query, request.explanation)

    return FlowchartResponse(
        diagram=diagram,
        success=diagram is not None,
    )

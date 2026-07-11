#!/bin/bash
echo ""
echo "  ████████╗██╗  ██╗██████╗ ██╗███████╗████████╗"
echo "     ██╔══╝██║  ██║██╔══██╗██║██╔════╝╚══██╔══╝"
echo "     ██║   ███████║██████╔╝██║█████╗     ██║   "
echo "     ██║   ██╔══██║██╔══██╗██║██╔══╝     ██║   "
echo "     ██║   ██║  ██║██║  ██║██║██║        ██║   "
echo "     ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝        ╚═╝   "
echo ""
echo "  Token Heuristic Routing with Intelligent Fallback Trees"
echo "  AMD Developer Hackathon ACT II — Track 1"
echo ""

if [ -z "$FW_API_KEY" ]; then
    echo "  ⚠  WARNING: FW_API_KEY not set. Tier 2 (Fireworks) will be unavailable."
    echo "     Set it with:  export FW_API_KEY=your-key-here"
    echo ""
fi

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "  ✓  Virtual environment activated."
fi

echo "  ✓  Starting THRIFT at http://localhost:8000"
echo "  ✓  Dashboard: http://localhost:8000"
echo "  ✓  API docs:  http://localhost:8000/docs"
echo "  Press Ctrl+C to stop."
echo ""

uvicorn app:app --host 0.0.0.0 --port 8000 --reload
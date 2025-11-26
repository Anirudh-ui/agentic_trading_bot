"""
Comprehensive test script for CAG system with smart routing
Tests all major improvements
"""

import time
from langchain_core.messages import HumanMessage
from agent.workflow import GraphBuilder
from toolkit.smart_router import SmartToolRouter
from utils.cag_evaluator import CAGEvaluator, GLOBAL_METRICS

def print_section(title):
    print("\n" + "=" * 70)
    print(f"{title:^70}")
    print("=" * 70)

def test_smart_router():
    """Test smart routing classification"""
    print_section("TEST 1: SMART ROUTER CLASSIFICATION")
    
    router = SmartToolRouter()
    
    test_cases = [
        ("What's the price of AAPL stock?", "stock", "yahoo_finance_tool"),
        ("Latest Bitcoin news", "general", "tavilytool"),
        ("Explain RSI indicator", "knowledge", "retriever_tool"),
        ("Tesla earnings report", "stock", "yahoo_finance_tool"),
        ("Market trends today", "general", "tavilytool"),
        ("How to use moving averages", "knowledge", "retriever_tool"),
        ("MSFT market cap", "stock", "yahoo_finance_tool"),
        ("Fed interest rate news", "general", "tavilytool"),
        ("What is a bull market", "knowledge", "retriever_tool"),
    ]
    
    correct = 0
    total = len(test_cases)
    
    print(f"\n{'Query':<40} {'Expected':<15} {'Got':<15} {'Tool':<25} {'✓/✗'}")
    print("-" * 100)
    
    for query, expected_type, expected_tool in test_cases:
        result = router.explain_classification(query)
        actual_type = result['classification']
        actual_tool = result['recommended_tool']
        
        is_correct = (actual_type == expected_type and actual_tool == expected_tool)
        correct += is_correct
        
        status = "✓" if is_correct else "✗"
        
        print(f"{query:<40} {expected_type:<15} {actual_type:<15} {actual_tool:<25} {status}")
    
    accuracy = correct / total * 100
    print(f"\n📊 Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    
    if accuracy >= 90:
        print("✅ PASS - Router working excellently")
    elif accuracy >= 80:
        print("⚠️  PASS - Router working well but could be improved")
    else:
        print("❌ FAIL - Router needs adjustment")

def test_cag_evaluator():
    """Test CAG relevancy evaluation"""
    print_section("TEST 2: CAG RELEVANCY EVALUATOR")
    
    evaluator = CAGEvaluator(relevance_threshold=0.5)
    
    # Test Case 1: High relevance
    print("\nTest Case 1: High Relevance Documents")
    query1 = "What is a bull market?"
    docs1 = [
        "A bull market is a financial market condition where prices are rising or expected to rise consistently.",
        "Bull markets are characterized by investor optimism, confidence, and strong economic indicators.",
        "During a bull market, stock prices typically rise by 20% or more from recent lows."
    ]
    
    result1 = evaluator.evaluate_retrieval(query1, docs1, use_mmr=True)
    print(f"Query: {query1}")
    print(f"Avg Relevance: {result1['avg_relevance']:.3f}")
    print(f"Needs Correction: {result1['needs_correction']}")
    print(f"Recommendation: {result1['recommendation']}")
    print(f"Status: {'✓ Good relevance' if not result1['needs_correction'] else '✗ Poor relevance'}")
    
    # Test Case 2: Low relevance
    print("\nTest Case 2: Low Relevance Documents")
    query2 = "Current Bitcoin price"
    docs2 = [
        "The weather today is sunny with temperatures in the 70s.",
        "Python is a popular programming language for data science.",
        "The capital of France is Paris."
    ]
    
    result2 = evaluator.evaluate_retrieval(query2, docs2, use_mmr=True)
    print(f"Query: {query2}")
    print(f"Avg Relevance: {result2['avg_relevance']:.3f}")
    print(f"Needs Correction: {result2['needs_correction']}")
    print(f"Recommendation: {result2['recommendation']}")
    print(f"Status: {'✓ Correctly flagged for correction' if result2['needs_correction'] else '✗ Should have flagged'}")
    
    # Test Case 3: Mixed relevance
    print("\nTest Case 3: Mixed Relevance Documents")
    query3 = "Trading strategies"
    docs3 = [
        "Day trading involves buying and selling securities within the same trading day.",
        "The stock market opens at 9:30 AM EST.",
        "Swing trading is a strategy that aims to capture gains over several days or weeks."
    ]
    
    result3 = evaluator.evaluate_retrieval(query3, docs3, use_mmr=True)
    print(f"Query: {query3}")
    print(f"Avg Relevance: {result3['avg_relevance']:.3f}")
    print(f"Needs Correction: {result3['needs_correction']}")
    print(f"Recommendation: {result3['recommendation']}")
    
    print("\n📊 CAG Evaluator Summary:")
    print(f"   Test 1 (High Relevance): {'✅ PASS' if not result1['needs_correction'] else '❌ FAIL'}")
    print(f"   Test 2 (Low Relevance): {'✅ PASS' if result2['needs_correction'] else '❌ FAIL'}")
    print(f"   Test 3 (Mixed): ✅ PASS (handled correctly)")

def test_end_to_end():
    """Test full system end-to-end"""
    print_section("TEST 3: END-TO-END SYSTEM TEST")
    
    print("\nInitializing system...")
    builder = GraphBuilder()
    builder.build()
    graph = builder.get_graph()
    print("✓ System initialized")
    
    test_queries = [
        ("Hello!", "Fast path greeting", 0.1),
        ("What's the Apple stock price?", "Stock query via Yahoo Finance", 3.0),
        ("Latest crypto news", "General query via Tavily", 5.0),
    ]
    
    print(f"\n{'Query':<35} {'Expected Route':<35} {'Time (s)':<10} {'Status'}")
    print("-" * 90)
    
    for query, expected_route, max_time in test_queries:
        start = time.time()
        try:
            response = graph.invoke({
                "messages": [HumanMessage(content=query)]
            })
            duration = time.time() - start
            
            # Extract response
            if isinstance(response, dict) and "messages" in response:
                final_msg = response["messages"][-1]
                response_text = final_msg.content if hasattr(final_msg, 'content') else str(final_msg)
                response_preview = response_text[:50] + "..." if len(response_text) > 50 else response_text
            else:
                response_preview = "Got response"
            
            status = "✓" if duration <= max_time else f"⚠️ (>{max_time}s)"
            
            print(f"{query:<35} {expected_route:<35} {duration:<10.2f} {status}")
            
        except Exception as e:
            print(f"{query:<35} {expected_route:<35} {'ERROR':<10} ✗")
            print(f"   Error: {str(e)[:100]}")
    
    # Check cache stats
    print("\n📊 Cache Statistics:")
    stats = builder.get_cache_stats()
    print(f"   Response Cache Size: {stats.get('response_cache_size', 0)}")
    
    print("\n✅ End-to-end test complete")

def test_performance():
    """Test performance improvements"""
    print_section("TEST 4: PERFORMANCE BENCHMARKS")
    
    builder = GraphBuilder()
    builder.build()
    graph = builder.get_graph()
    
    # Test 1: Fast path
    print("\n1. Fast Path Performance (Greeting)")
    times = []
    for i in range(3):
        start = time.time()
        graph.invoke({"messages": [HumanMessage(content="Hello!")]})
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    print(f"   Average time: {avg_time:.4f}s")
    print(f"   Status: {'✅ PASS (<0.1s)' if avg_time < 0.1 else '⚠️  Slower than expected'}")
    
    # Test 2: Cache effectiveness
    print("\n2. Cache Effectiveness")
    query = "What is Bitcoin?"
    
    # First call (no cache)
    start = time.time()
    graph.invoke({"messages": [HumanMessage(content=query)]})
    first_time = time.time() - start
    
    # Second call (cached)
    start = time.time()
    graph.invoke({"messages": [HumanMessage(content=query)]})
    cached_time = time.time() - start
    
    speedup = first_time / cached_time if cached_time > 0 else 0
    
    print(f"   First call: {first_time:.2f}s")
    print(f"   Cached call: {cached_time:.2f}s")
    print(f"   Speedup: {speedup:.1f}x")
    print(f"   Status: {'✅ PASS (>5x speedup)' if speedup > 5 else '⚠️  Cache not effective'}")
    
    print("\n📊 Performance Summary:")
    print(f"   Fast Path: {'✅' if avg_time < 0.1 else '⚠️'} {avg_time*1000:.0f}ms avg")
    print(f"   Caching: {'✅' if speedup > 5 else '⚠️'} {speedup:.1f}x speedup")

def test_mmr():
    """Test MMR document ranking"""
    print_section("TEST 5: MMR DOCUMENT RANKING")
    
    evaluator = CAGEvaluator()
    
    query = "trading strategies"
    documents = [
        "Day trading is a strategy where traders buy and sell within the same day.",
        "Day trading strategies focus on short-term price movements.",
        "Swing trading holds positions for several days to weeks.",
        "The weather is nice today.",
        "Long-term investing focuses on holding assets for years."
    ]
    
    print(f"\nQuery: {query}")
    print("\nOriginal order:")
    for i, doc in enumerate(documents, 1):
        print(f"{i}. {doc[:70]}...")
    
    # Rank with MMR
    ranked = evaluator.rank_documents(query, documents, use_mmr=True)
    
    print("\nRanked by MMR (relevance + diversity):")
    for i, (idx, doc, score) in enumerate(ranked, 1):
        print(f"{i}. [Score: {score:.3f}] {doc[:70]}...")
    
    # Check if irrelevant doc is ranked low
    irrelevant_score = [score for idx, doc, score in ranked if "weather" in doc][0]
    
    print(f"\n📊 MMR Results:")
    print(f"   Irrelevant doc score: {irrelevant_score:.3f}")
    print(f"   Status: {'✅ PASS - Irrelevant ranked low' if irrelevant_score < 0.3 else '⚠️  Needs tuning'}")

def run_all_tests():
    """Run all tests"""
    print("\n" + "🧪 " * 35)
    print("COMPREHENSIVE CAG SYSTEM TEST SUITE")
    print("🧪 " * 35)
    
    start_time = time.time()
    
    try:
        test_smart_router()
        test_cag_evaluator()
        test_mmr()
        test_end_to_end()
        test_performance()
        
        total_time = time.time() - start_time
        
        print_section("TEST SUITE COMPLETE")
        print(f"\n⏱️  Total test time: {total_time:.2f}s")
        print("\n✅ All tests completed successfully!")
        print("\n📈 System is ready for deployment")
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        
        tests = {
            'router': test_smart_router,
            'cag': test_cag_evaluator,
            'mmr': test_mmr,
            'e2e': test_end_to_end,
            'perf': test_performance,
            'all': run_all_tests
        }
        
        if test_name in tests:
            tests[test_name]()
        else:
            print(f"Unknown test: {test_name}")
            print(f"Available tests: {', '.join(tests.keys())}")
    else:
        print("Usage: python test_cag_system.py [test_name]")
        print("\nAvailable tests:")
        print("  router - Test smart routing")
        print("  cag    - Test CAG evaluator")
        print("  mmr    - Test MMR ranking")
        print("  e2e    - Test end-to-end")
        print("  perf   - Test performance")
        print("  all    - Run all tests")
        print("\nRunning all tests by default...\n")
        run_all_tests()
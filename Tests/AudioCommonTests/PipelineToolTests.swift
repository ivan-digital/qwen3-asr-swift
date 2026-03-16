import XCTest
@testable import AudioCommon

final class PipelineToolTests: XCTestCase {

    func testToolCreation() {
        let tool = PipelineTool(name: "get_time", description: "Returns current time") { _ in
            "12:00"
        }
        XCTAssertEqual(tool.name, "get_time")
        XCTAssertEqual(tool.description, "Returns current time")
        XCTAssertEqual(tool.cooldown, 0)
    }

    func testToolWithCooldown() {
        let tool = PipelineTool(name: "search", description: "Web search", cooldown: 30) { query in
            "Results for: \(query)"
        }
        XCTAssertEqual(tool.cooldown, 30)
    }

    func testToolHandlerExecution() {
        var called = false
        let tool = PipelineTool(name: "test", description: "test") { args in
            called = true
            return "got: \(args)"
        }
        let result = tool.handler("{\"query\": \"hello\"}")
        XCTAssertTrue(called)
        XCTAssertEqual(result, "got: {\"query\": \"hello\"}")
    }

    func testToolHandlerEmptyArgs() {
        let tool = PipelineTool(name: "no_args", description: "No args tool") { args in
            XCTAssertEqual(args, "")
            return "done"
        }
        XCTAssertEqual(tool.handler(""), "done")
    }

    func testMultipleToolsCreation() {
        let tools = [
            PipelineTool(name: "a", description: "Tool A") { _ in "a" },
            PipelineTool(name: "b", description: "Tool B") { _ in "b" },
            PipelineTool(name: "c", description: "Tool C", cooldown: 10) { _ in "c" },
        ]
        XCTAssertEqual(tools.count, 3)
        XCTAssertEqual(tools[0].name, "a")
        XCTAssertEqual(tools[2].cooldown, 10)
    }
}

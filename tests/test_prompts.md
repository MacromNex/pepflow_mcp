# MCP Integration Test Prompts

This file contains systematic test prompts for validating the optimization-analysis MCP server integration with Claude Code.

## Test Information
- **Server Name**: optimization-analysis
- **Tools Available**: 15 total (5 job management, 3 sync analysis, 4 submit analysis, 3 utilities)
- **Test Environment**: Claude Code with MCP integration
- **Test Date**: 2024-12-31

## Tool Discovery Tests

### Prompt 1: List All Tools
**Test Purpose**: Verify tool discovery and access
**Expected Outcome**: Should list all 15 tools with descriptions

```
What MCP tools are available for optimization analysis? Give me a brief description of each tool and categorize them by type.
```

### Prompt 2: Tool Details
**Test Purpose**: Verify tool introspection
**Expected Outcome**: Should provide detailed parameter information

```
Explain how to use the analyze_gradient_descent tool, including all parameters, default values, and what type of results it returns.
```

### Prompt 3: Server Information
**Test Purpose**: Verify server metadata access
**Expected Outcome**: Should identify server name and capabilities

```
What optimization analysis server is connected? What algorithms and frameworks does it support?
```

---

## Synchronous Tool Tests

### Prompt 4: Basic Gradient Descent Analysis
**Test Purpose**: Test sync tool with default parameters
**Expected Runtime**: < 30 seconds
**Expected Outcome**: Convergence analysis with O(1/k) verification

```
Run a basic gradient descent convergence analysis using the default parameters. Show me the convergence rate and verification results.
```

### Prompt 5: Custom Gradient Descent Parameters
**Test Purpose**: Test sync tool with custom parameters
**Expected Runtime**: < 60 seconds
**Expected Outcome**: Analysis with specified parameters

```
Analyze gradient descent convergence with 12 iterations, 3 proof steps, Lipschitz constant of 2.0, and initial radius of 0.5.
```

### Prompt 6: Accelerated Gradient Analysis
**Test Purpose**: Test AGM sync tool
**Expected Runtime**: < 45 seconds
**Expected Outcome**: O(1/k²) convergence analysis

```
Run an accelerated gradient method analysis with 8 iterations and 5 proof steps. Explain the theta sequence computation and convergence rate.
```

### Prompt 7: Algorithm Comparison
**Test Purpose**: Test multi-algorithm analysis
**Expected Runtime**: < 90 seconds
**Expected Outcome**: Comparison of optimization methods

```
Compare the performance of gradient descent, heavy ball, and accelerated gradient methods with 6 iterations each and momentum parameter 0.3 for heavy ball.
```

### Prompt 8: Error Handling - Invalid Parameters
**Test Purpose**: Test error handling in sync tools
**Expected Outcome**: Structured error message with helpful guidance

```
Run gradient descent analysis with invalid parameters: iterations=-5, proof_steps=0, lipschitz_constant=-1.0.
```

---

## Submit API and Job Management Tests

### Prompt 9: Submit Long-Running Job
**Test Purpose**: Test async job submission
**Expected Outcome**: Job ID and tracking instructions

```
Submit a large-scale gradient descent analysis job with 25 iterations and 15 proof steps. This should run in the background.
```

### Prompt 10: Check Job Status
**Test Purpose**: Test job status tracking
**Expected Outcome**: Job status with timestamps
**Note**: Use job_id from previous test

```
Check the status of job <job_id_from_prompt_9>. Show me the current state and any progress information.
```

### Prompt 11: Monitor Job Logs
**Test Purpose**: Test job log access
**Expected Outcome**: Recent log lines from running job

```
Show me the last 20 lines of logs for job <job_id_from_prompt_9> to see what it's currently processing.
```

### Prompt 12: List All Jobs
**Test Purpose**: Test job listing functionality
**Expected Outcome**: List of all submitted jobs with status

```
List all optimization analysis jobs that have been submitted, showing their status and submission times.
```

### Prompt 13: Submit Multiple Jobs
**Test Purpose**: Test concurrent job handling
**Expected Outcome**: Multiple job IDs

```
Submit three different analysis jobs:
1. Accelerated gradient analysis with 20 iterations
2. Optimization comparison with 15 iterations
3. Parameter sweep for gradient descent
Track all the job IDs returned.
```

### Prompt 14: Get Job Results
**Test Purpose**: Test result retrieval from completed job
**Expected Outcome**: Structured analysis results
**Note**: Use job_id from completed job

```
Get the detailed results from the completed job <completed_job_id>. Show me the convergence analysis findings.
```

### Prompt 15: Cancel Running Job
**Test Purpose**: Test job cancellation
**Expected Outcome**: Confirmation of cancellation
**Note**: Use job_id from running job

```
Cancel the running job <running_job_id> and confirm the cancellation was successful.
```

---

## Batch Processing Tests

### Prompt 16: Parameter Sweep Analysis
**Test Purpose**: Test batch parameter analysis
**Expected Outcome**: Job submission for parameter sweep

```
Submit a parameter sweep analysis for the heavy ball method, testing Lipschitz constants [0.5, 1.0, 1.5, 2.0] and momentum values [0.1, 0.3, 0.5, 0.7]. Use 8 iterations for each combination.
```

### Prompt 17: Algorithm Comparison Batch
**Test Purpose**: Test large-scale algorithm comparison
**Expected Outcome**: Comprehensive comparison job

```
Submit a comprehensive optimization algorithm comparison job that analyzes gradient descent, heavy ball, and accelerated gradient methods with 12 iterations each. Include performance metrics for each method.
```

---

## Utility and Information Tests

### Prompt 18: Available Algorithms
**Test Purpose**: Test algorithm discovery
**Expected Outcome**: List of supported algorithms with properties

```
What optimization algorithms are available for analysis? Show me their convergence rates, parameters, and descriptions.
```

### Prompt 19: Default Configurations
**Test Purpose**: Test configuration access
**Expected Outcome**: Default config parameters for all algorithms

```
Show me the default configuration parameters for all optimization algorithms, including iterations, proof steps, and algorithmic parameters.
```

### Prompt 20: Framework Validation
**Test Purpose**: Test installation validation
**Expected Outcome**: PEPFlow framework status and version info

```
Validate that the PEPFlow framework is properly installed and show me the version information for all dependencies.
```

---

## End-to-End Scenarios

### Prompt 21: Complete Analysis Workflow
**Test Purpose**: Test full analysis pipeline
**Expected Outcome**: Complete analysis from submission to results

```
Perform a complete optimization analysis workflow:
1. First validate the PEPFlow installation
2. Get the default configuration for gradient descent
3. Submit a gradient descent analysis job with 20 iterations
4. Monitor the job status until completion
5. Retrieve and summarize the final results
```

### Prompt 22: Algorithm Research Scenario
**Test Purpose**: Test research workflow
**Expected Outcome**: Comparative analysis results

```
I'm researching convergence rates for my optimization project. Help me:
1. List all available algorithms with their theoretical convergence rates
2. Run quick synchronous analyses for gradient descent and accelerated gradient (5 iterations each)
3. Submit a detailed comparison job for longer analysis (15 iterations)
4. Compare the theoretical vs. empirical convergence results
```

### Prompt 23: Parameter Optimization Scenario
**Test Purpose**: Test parameter tuning workflow
**Expected Outcome**: Parameter recommendations

```
I want to optimize parameters for the heavy ball method. Help me:
1. Get the default heavy ball configuration
2. Submit a parameter sweep for momentum values [0.2, 0.4, 0.6, 0.8]
3. Monitor the sweep job progress
4. Analyze results to recommend the best momentum parameter
```

### Prompt 24: Performance Validation Scenario
**Test Purpose**: Test validation workflow
**Expected Outcome**: Performance verification

```
Validate the performance claims for accelerated gradient method:
1. Run a quick accelerated gradient analysis (sync)
2. Submit a detailed accelerated gradient job (25 iterations, 12 proof steps)
3. Compare with basic gradient descent performance
4. Verify the O(1/k²) vs O(1/k) convergence rate difference
```

### Prompt 25: Error Recovery Scenario
**Test Purpose**: Test error handling and recovery
**Expected Outcome**: Graceful error handling and helpful guidance

```
Test error handling by:
1. Attempting analysis with invalid algorithm name "nonexistent_method"
2. Submitting a job with impossible parameters (negative iterations)
3. Trying to get results for a non-existent job ID "fake123"
4. Show how the system handles each error case
```

---

## Expected Success Criteria

### Tool Discovery (Prompts 1-3)
- ✅ All 15 tools listed correctly
- ✅ Tool descriptions accurate and helpful
- ✅ Server identification correct

### Synchronous Tools (Prompts 4-8)
- ✅ All sync tools execute within 2 minutes
- ✅ Results returned in structured format
- ✅ Mathematical analysis appears correct
- ✅ Error handling provides helpful messages

### Submit API (Prompts 9-15)
- ✅ Jobs submit successfully with valid job IDs
- ✅ Status tracking works throughout lifecycle
- ✅ Log access provides real-time information
- ✅ Job listing shows all submitted jobs
- ✅ Result retrieval works for completed jobs
- ✅ Job cancellation works for running jobs

### Batch Processing (Prompts 16-17)
- ✅ Parameter sweep jobs submit successfully
- ✅ Complex comparison jobs handle multiple algorithms
- ✅ Batch results accessible when completed

### Utilities (Prompts 18-20)
- ✅ Algorithm information complete and accurate
- ✅ Configuration data properly formatted
- ✅ Framework validation confirms installation

### End-to-End (Prompts 21-25)
- ✅ Complete workflows execute successfully
- ✅ Research scenarios provide useful insights
- ✅ Parameter optimization yields recommendations
- ✅ Performance validation confirms theoretical claims
- ✅ Error scenarios handled gracefully

---

## Notes for Testing

1. **Timing**: Record actual execution times for sync tools
2. **Job IDs**: Track job IDs from submit operations for status/result checks
3. **Error Cases**: Verify error messages are helpful and actionable
4. **Resource Usage**: Monitor system resources during large jobs
5. **Concurrency**: Test multiple simultaneous job submissions
6. **Recovery**: Test server restart scenarios if possible

---

## Test Environment Setup

Before running tests, ensure:
- ✅ MCP server registered in Claude Code: `claude mcp list`
- ✅ Server health check passes: optimization-analysis connected
- ✅ PEPFlow framework installed in project environment
- ✅ Job directory exists and is writable: `./jobs/`
- ✅ All dependencies available: numpy, sympy, matplotlib, etc.
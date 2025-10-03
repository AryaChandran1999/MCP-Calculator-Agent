import os
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import asyncio
from google import genai
from concurrent.futures import TimeoutError
from threading import Thread
import math

# Load environment variables from .env file
load_dotenv()

# Access your API key and initialize Gemini client correctly
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

max_iterations = 5
last_response = None
iteration = 0
iteration_response = []

async def generate_with_timeout(client, prompt, timeout=10):
    """Generate content with a timeout"""
    print("Starting LLM generation...")
    try:
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
            ),
            timeout=timeout
        )
        print("LLM generation completed")
        return response
    except TimeoutError:
        print("LLM generation timed out!")
        raise
    except Exception as e:
        print(f"Error in LLM generation: {e}")
        raise

def reset_state():
    """Reset all global variables to their initial state"""
    global last_response, iteration, iteration_response
    last_response = None
    iteration = 0
    iteration_response = []

async def main():
    reset_state()
    print("Starting main execution...")
    try:
        # MCP server connection
        print("Establishing connection to MCP server...")
        server_params = StdioServerParameters(
            command="python",
            args=["example2_gmail.py"]
        )

        async with stdio_client(server_params) as (read, write):
            print("Connection established, creating session...")
            async with ClientSession(read, write) as session:
                print("Session created, initializing...")
                await session.initialize()

                # Get available tools
                print("Requesting tool list...")
                tools_result = await session.list_tools()
                tools = tools_result.tools
                print(f"Successfully retrieved {len(tools)} tools")

                # Create system prompt
                tools_description = []
                for i, tool in enumerate(tools):
                    try:
                        params = tool.inputSchema
                        desc = getattr(tool, 'description', 'No description available')
                        name = getattr(tool, 'name', f'tool_{i}')

                        if 'properties' in params:
                            param_details = []
                            for param_name, param_info in params['properties'].items():
                                param_type = param_info.get('type', 'unknown')
                                param_details.append(f"{param_name}: {param_type}")
                            params_str = ', '.join(param_details)
                        else:
                            params_str = 'no parameters'

                        tool_desc = f"{i+1}. {name}({params_str}) - {desc}"
                        tools_description.append(tool_desc)
                        print(f"Added description for tool: {tool_desc}")
                    except Exception as e:
                        print(f"Error processing tool {i}: {e}")
                        tools_description.append(f"{i+1}. Error processing tool")
                
                tools_description = "\n".join(tools_description)
                print("Successfully created tools description")

                system_prompt = f"""You are a math agent solving problems in iterations. You have access to various mathematical tools.

Available tools:
{tools_description}

You must respond with EXACTLY ONE line in one of these formats (no additional text):
1. For function calls:
   FUNCTION_CALL: function_name|param1|param2|...
   
2. For final answers:
   FINAL_ANSWER: [number]

Important:
- When a function returns multiple values, you need to process all of them
- Only give FINAL_ANSWER when you have completed all necessary calculations
- After computing the numeric answer, output a tool call using: FUNCTION_CALL: send_answer_email|[numeric answer]
- Do not repeat function calls with the same parameters

Examples:
- FUNCTION_CALL: add|5|3
- FUNCTION_CALL: strings_to_chars_to_int|INDIA
- FINAL_ANSWER: [42] 
- FUNCTION_CALL: send_answer_email|42 #Called after the FINAL_ANSWER

DO NOT include any explanations or additional text.
Your entire response should be a single line starting with either FUNCTION_CALL: or FINAL_ANSWER:"""

                query = """Find the ASCII values of characters in INDIA and then return sum of exponentials of those values."""
                global iteration, last_response

                while iteration < max_iterations:
                    print(f"\n--- Iteration {iteration + 1} ---")
                    if last_response is None:
                        current_query = query
                    else:
                        current_query = current_query + "\n\n" + " ".join(iteration_response)
                        current_query = current_query + "  What should I do next?"

                    # Generate LLM response
                    prompt = f"{system_prompt}\n\nQuery: {current_query}"
                    try:
                        response = await generate_with_timeout(client, prompt)
                        response_text = response.text.strip()
                        print(f"LLM Response: {response_text}")

                        # Keep only the first FUNCTION_CALL or FINAL_ANSWER line
                        for line in response_text.split('\n'):
                            line = line.strip()
                            if line.startswith(("FUNCTION_CALL:", "FINAL_ANSWER:")):
                                response_text = line
                                break
                    except Exception as e:
                        print(f"Failed to get LLM response: {e}")
                        break

                    if response_text.startswith("FUNCTION_CALL:"):
                        _, function_info = response_text.split(":", 1)
                        parts = [p.strip() for p in function_info.split("|")]
                        func_name, params = parts[0], parts[1:]

                        try:
                            tool = next((t for t in tools if t.name == func_name), None)
                            if not tool:
                                raise ValueError(f"Unknown tool: {func_name}")

                            arguments = {}
                            schema_properties = tool.inputSchema.get('properties', {})
                            for param_name, param_info in schema_properties.items():
                                if not params:
                                    raise ValueError(f"Not enough parameters provided for {func_name}")
                                value = params.pop(0)
                                param_type = param_info.get('type', 'string')
                                if param_type == 'integer':
                                    arguments[param_name] = int(value)
                                elif param_type == 'number':
                                    arguments[param_name] = float(value)
                                elif param_type == 'array':
                                    if isinstance(value, str):
                                        value = value.strip('[]').split(',')
                                    arguments[param_name] = [int(x.strip().strip("'").strip('"')) for x in value]
                                else:
                                    arguments[param_name] = str(value)

                            result = await session.call_tool(func_name, arguments=arguments)
                            if hasattr(result, 'content') and isinstance(result.content, list):
                                iteration_result = [
                                    item.text if hasattr(item, 'text') else str(item)
                                    for item in result.content
                                ]
                            else:
                                iteration_result = str(result)

                            iteration_response.append(
                                f"In the {iteration + 1} iteration you called {func_name} with {arguments} parameters, "
                                f"and the function returned {iteration_result}."
                            )
                            last_response = iteration_result
                            print(f"Last response: {last_response}")

                        except Exception as e:
                            print(f"Error in iteration {iteration + 1}: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            iteration_response.append(f"Error in iteration {iteration + 1}: {str(e)}")
                            break

                    elif response_text.startswith("FINAL_ANSWER:"):
                        print("\n=== Agent Execution Complete ===")
                        print(f"Agent gave FINAL_ANSWER: {response_text}")
                        iteration_response.append(f"FINAL_ANSWER received: {response_text}")
                        # break
                        last_response = response_text
                        print(f"Last response: {last_response}")

                    iteration += 1

    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        reset_state()

if __name__ == "__main__":
    asyncio.run(main())

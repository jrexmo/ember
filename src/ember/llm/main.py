import click
import openai
from rich.console import Console
from rich.json import JSON as RichJSON
import yaml
from dotenv import load_dotenv
import os
import json
import subprocess
from rich.console import Console
from rich.markdown import Markdown

# Load environment variables from .env file
load_dotenv()
load_dotenv(".env.secret")

# Retrieve API key from environment variables
API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = openai.OpenAI(api_key=API_KEY)


# Load the system prompt from the YAML file
def load_system_prompt(filepath):
    with open(filepath, "r") as file:
        return yaml.safe_load(file)["system_prompt"]


# Define functions to be used with the model
def run_ls(path="."):
    """Simulate the `ls` command."""
    try:
        result = subprocess.run(
            ["ls", path], capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return str(e)


def run_cat(path):
    """Simulate the `cat` command."""
    try:
        result = subprocess.run(
            ["cat", path], capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return str(e)


def edit_file(path, content):
    """Edit the contents of a file."""
    try:
        with open(path, "w") as file:
            file.write(content)
        return f"File '{path}' updated successfully."
    except Exception as e:
        return str(e)


# Update available functions
available_functions = {
    "ls": run_ls,
    "cat": run_cat,
    "edit_file": edit_file,  # Add the `edit_file` function here
}


# Define the feature development prompt
def load_feature_request_prompt(filepath):
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


# Process user requests
def handle_feature_request(feature_request, system_prompt_path, user_prompt_path):
    console = Console()
    prompt = load_feature_request_prompt(user_prompt_path)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "ls",
                "description": "List files in a directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The directory path to list.",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cat",
                "description": "Read the contents of a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The file path to read.",
                        },
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Edit the contents of a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The file path to edit.",
                        },
                        "content": {
                            "type": "string",
                            "description": "The new content to write to the file.",
                        },
                    },
                    "required": ["path", "content"],
                },
            },
        },
    ]

    messages = [
        {"role": "system", "content": prompt["system_prompt"]},
        {
            "role": "user",
            "content": prompt["user_prompt"].format(feature_request=feature_request),
        },
    ]
    # Initial request to the model
    console.print("Initial request to the model:", style="bold green")
    console.print(messages)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    response_message = response.choices[0].message

    console.print("Initial model response:", style="bold green")
    console.print(response_message, style="cyan")

    # Process tool calls
    _exit = False
    while not _exit and response_message.tool_calls:
        messages.append(response_message)
        console.print(messages)
        console.print("Model function calls:", style="bold yellow")
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            function_to_call = available_functions[function_name]
            function_response = function_to_call(**function_args)

            console.print(f"Tool call ID: {tool_call.id}", style="magenta")
            console.print(f"Function name: {function_name}", style="magenta")
            console.print("Function arguments:", style="magenta")
            console.print(RichJSON(tool_call.function.arguments), style="blue")
            console.print(f"Function output for {function_name}:", style="bold yellow")
            console.print(function_response, style="green")

            # Add function call and response to messages
            messages.append(
                {
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                    "tool_call_id": tool_call.id,
                }
            )

        # Get a new response from the model
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=tools
        )
        response_message = response.choices[0].message

    messages.append(response_message)
    console.print(messages)

    # Print the summary with markdown rendering
    summary = response_message.content
    console.print("Final summary:", style="bold green")
    # markdown = Markdown(summary)
    console.print(summary)


def summarize_content(content, model):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an intelligent assistant specialized in summarizing text. Provide concise and clear summaries of the input text.",
            },
            {"role": "user", "content": content},
        ],
    )
    return completion.choices[0].message.content


@click.group()
def cli():
    pass


@cli.command()
@click.argument("file", type=click.Path(exists=True, readable=True))
@click.option(
    "--system-prompt",
    default="src/ember/system_prompt.yaml",
    help="Path to the YAML system prompt file.",
)
@click.option(
    "--model", default="gpt-4o-mini", help="The model to use for summarization."
)
def summarize(file, system_prompt, model):
    """Summarize the content of a file."""
    # Read file content
    with open(file, "r") as f:
        content = f.read()

    # Get the summary
    summary = summarize_content(content, model)

    # Print the summary with markdown rendering
    console = Console()
    markdown = Markdown(summary)
    console.print(markdown)


@cli.command()
@click.argument("feature_request")
@click.option(
    "--system-prompt",
    default="src/ember/llm/system_prompt.yaml",
    help="Path to the YAML system prompt file.",
)
@click.option(
    "--user-prompt",
    default="src/ember/llm/feature_request_prompt.yaml",
    help="Path to the YAML feature request prompt file.",
)
def add_feature(feature_request, system_prompt, user_prompt):
    """Handle a feature request and suggest changes or new files."""
    handle_feature_request(feature_request, system_prompt, user_prompt)


if __name__ == "__main__":
    cli()

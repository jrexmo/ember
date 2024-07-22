# Ember

A tool for creating embroidery patterns from images.

## Project Structure

```
./src:
  ember - application directory

./src/ember:
  embroidery.py - embroidery operations
  utils.py - utility functions
```

## Development Tools

- Python 3.12
- [Rye](https://rye.astral.sh/) - Installation: `curl -sSf https://rye.astral.sh/get | bash`

## Usage

1. Clone the repository:
    ```sh
    git clone https://github.com/thornewolf/ember
    cd ember
    ```

2. Install dependencies:
    ```sh
    rye sync
    ```

3. Prepare the image:
    ```sh
    mkdir data
    wget <image> -O data/image.jpg  # or place an image in the data directory manually
    ```

4. Run the script:
    ```sh
    rye run python src/ember/embroidery.py
    ```

## LLM Tool Usage

The `llm` tool allows you to interact with the OpenAI language model to summarize content and handle feature requests.

### Commands

1. **Summarize a File**
   To summarize a file, use the following command:
   ```sh
   rye run python src/ember/llm/main.py summarize <path_to_your_file>
   ```

   Options:
   - `--system-prompt`: Specify the path to the YAML system prompt file. Default is `src/ember/system_prompt.yaml`.
   - `--model`: Specify the model to use for summarization. Default is `gpt-4o-mini`.

   Example:
   ```sh
   rye run python src/ember/llm/main.py summarize README.md
   ```

2. **Handle a Feature Request**
   To handle feature requests, use:
   ```sh
   rye run python src/ember/llm/main.py add_feature "<feature_request>"
   ```

   Options:
   - `--system-prompt`: Specify the path to the YAML system prompt file. Default is `src/ember/llm/system_prompt.yaml`.
   - `--user-prompt`: Specify the path to the YAML feature request prompt file. Default is `src/ember/llm/feature_request_prompt.yaml`.

   Example:
   ```sh
   rye run python src/ember/llm/main.py add_feature "Add support for creating SVG files."
   ```

## TODO

### Functionality
- [x] Repository setup
- [ ] Fix contour-to-stitch functionality. Ensure stitches are closed by adding a connecting stitch between the first and last contour points.
- [ ] Implement contour filling. Identify "islands" of pixels and fill them using DFS, prioritizing movements in the order: <move same direction, move right, move left, move down>. Perform DFS from top left to bottom right to ensure proper filling.
- [ ] Define background contour to address any issues with unfilled areas.

### Server
- [ ] Create a FastAPI interface for `embroidery.py`.

### Tooling
- [ ] Develop a CLI interface for `embroidery.py`.
- [ ] Create a `Justfile` for project automation.

## References
- [Zee Digitizing](https://zeedigitizing.com/)
- [Graphic Design Stack Exchange](https://graphicdesign.stackexchange.com/questions/66375/how-can-i-digitize-my-vector-to-a-dst-for-embroidery-for-free)
- [Reddit - Design](https://www.reddit.com/r/Design/comments/150429b/convert_vector_files_to_dst/)
- [Etsy - Digitizing](https://www.etsy.com/market/new_york_digitizing?ref=lp_queries_external_top-2)
- [DST Format GitHub](https://github.com/manthrax/dst-format)
- [pyembroidery GitHub](https://github.com/EmbroidePy/pyembroidery/tree/main)
- [Girl Face Embroidery PDF](https://zdigitizing.net/wp-content/uploads/2023/08/Girl-Face-4x4-1.pdf)

## Documentation
[Project Documentation](https://drive.google.com/drive/folders/12mpAu_EJxsFcHfvugEbbVUF2j9kYIurb)

### Additional Dependencies
- [JinjaX](https://jinjax.scaletti.dev/)

## Collaborators
- Thorne Wolfenbarger
- Daniel Bump
- Jackson Morris
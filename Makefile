# Global Dialogues Analysis Makefile

# Variables
PYTHON := python
TOOLS_DIR := tools/scripts
ANALYSIS_DIR := tools/analysis

# Colors for terminal output
BLUE := \033[34m
GREEN := \033[32m
RED := \033[31m
YELLOW := \033[33m
RESET := \033[0m

.PHONY: help preprocess analyze clean \
        preprocess-gd1 preprocess-gd2 preprocess-gd3 preprocess-all \
        analyze-gd1 analyze-gd2 analyze-gd3 analyze-all \
        tag-analysis preprocess-tags \
        download-embeddings download-embeddings-gd1 download-embeddings-gd2 download-embeddings-gd3 download-all-embeddings \
        run-thematic-ranking run-thematic-ranking-gd1 run-thematic-ranking-gd2 run-thematic-ranking-gd3 \
        pri pri-llm pri-gd1 pri-gd2 pri-gd3 pri-llm-gd1 pri-llm-gd2 pri-llm-gd3

# Default target
help:
	@echo "$(BLUE)Global Dialogues Analysis Commands:$(RESET)"
	@echo "  $(GREEN)make help$(RESET)                 - Show this help message"
	@echo ""
	@echo "$(BLUE)Preprocessing Commands:$(RESET)"
	@echo "  $(GREEN)make preprocess-gd1$(RESET)       - Preprocess GD1 data (metadata cleanup + standardize aggregate)"
	@echo "  $(GREEN)make preprocess-gd2$(RESET)       - Preprocess GD2 data (metadata cleanup + standardize aggregate)"
	@echo "  $(GREEN)make preprocess-gd3$(RESET)       - Preprocess GD3 data (metadata cleanup + standardize aggregate)"
	@echo "  $(GREEN)make preprocess-all$(RESET)       - Preprocess all GD datasets"
	@echo "  $(GREEN)make preprocess-tags-gd1$(RESET)  - Preprocess tag data for GD1"
	@echo "  $(GREEN)make preprocess-tags-gd2$(RESET)  - Preprocess tag data for GD2"
	@echo "  $(GREEN)make preprocess-tags-gd3$(RESET)  - Preprocess tag data for GD3"
	@echo ""
	@echo "$(BLUE)Data Commands:$(RESET)"
	@echo "  $(GREEN)make download-embeddings$(RESET)  - Show available embedding files and download options"
	@echo "  $(GREEN)make download-embeddings-gd1$(RESET) - Download embeddings for GD1"
	@echo "  $(GREEN)make download-embeddings-gd2$(RESET) - Download embeddings for GD2" 
	@echo "  $(GREEN)make download-embeddings-gd3$(RESET) - Download embeddings for GD3"
	@echo "  $(GREEN)make download-all-embeddings$(RESET) - Download all embeddings"
	@echo ""
	@echo "$(BLUE)Analysis Commands:$(RESET)"
	@echo "  $(GREEN)make analyze-gd1$(RESET)          - Run full analysis pipeline on GD1"
	@echo "  $(GREEN)make analyze-gd2$(RESET)          - Run full analysis pipeline on GD2"
	@echo "  $(GREEN)make analyze-gd3$(RESET)          - Run full analysis pipeline on GD3"
	@echo "  $(GREEN)make analyze-all$(RESET)          - Run full analysis pipeline on all GD datasets"
	@echo ""
	@echo "$(BLUE)Individual Analysis Commands:$(RESET)"
	@echo "  $(GREEN)make consensus-gd<N>$(RESET)      - Calculate consensus metrics for GD<N>"
	@echo "  $(GREEN)make divergence-gd<N>$(RESET)     - Calculate divergence metrics for GD<N>"
	@echo "  $(GREEN)make indicators-gd<N>$(RESET)     - Generate indicator heatmaps for GD<N>"
	@echo "  $(GREEN)make tags-gd<N>$(RESET)           - Analyze tags for GD<N>"
	@echo ""
	@echo "$(BLUE)PRI (Participant Reliability Index) Commands:$(RESET)"
	@echo "  $(GREEN)make pri GD=<N>$(RESET)           - Calculate PRI for GD<N> (traditional metrics only)"
	@echo "  $(GREEN)make pri-llm GD=<N>$(RESET)       - Calculate PRI for GD<N> with LLM judge assessment"
	@echo "  $(GREEN)make pri-gd<N>$(RESET)            - Calculate PRI for specific GD (traditional metrics only)"
	@echo "  $(GREEN)make pri-llm-gd<N>$(RESET)        - Calculate PRI for specific GD with LLM judge"
	@echo ""
	@echo "$(BLUE)Advanced Analysis Commands:$(RESET)"
	@echo "  $(GREEN)make run-thematic-ranking$(RESET) - Run thematic ranking analysis (requires API key and embeddings)"
	@echo "  $(GREEN)make run-thematic-ranking-gd<N>$(RESET) - Run thematic ranking for GD<N>"
	@echo ""
	@echo "$(BLUE)Utilities:$(RESET)"
	@echo "  $(GREEN)make preview-csvs-gd<N>$(RESET)   - Preview all CSV files in GD<N> directory"
	@echo "  $(GREEN)make clean$(RESET)                - Clean up cache and temporary files"

# Preprocessing commands (metadata cleanup + standardize aggregate)
preprocess-gd1:
	@echo "$(BLUE)Preprocessing GD1 data...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/preprocess_cleanup_metadata.py --gd_number 1
	$(PYTHON) $(TOOLS_DIR)/preprocess_aggregate.py --gd_number 1

preprocess-gd2:
	@echo "$(BLUE)Preprocessing GD2 data...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/preprocess_cleanup_metadata.py --gd_number 2
	$(PYTHON) $(TOOLS_DIR)/preprocess_aggregate.py --gd_number 2

preprocess-gd3:
	@echo "$(BLUE)Preprocessing GD3 data...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/preprocess_cleanup_metadata.py --gd_number 3
	$(PYTHON) $(TOOLS_DIR)/preprocess_aggregate.py --gd_number 3

preprocess-all: preprocess-gd1 preprocess-gd2 preprocess-gd3
	@echo "$(GREEN)All datasets preprocessed successfully!$(RESET)"

# Tag preprocessing
preprocess-tags-gd1:
	@echo "$(BLUE)Preprocessing GD1 tag data...$(RESET)"
	@echo "$(YELLOW)NOTE: This requires raw tag exports in Data/GD1/tag_codes_raw/$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/preprocess_tag_files.py --raw_dir Data/GD1/tag_codes_raw/ --output_dir Data/GD1/tags/

preprocess-tags-gd2:
	@echo "$(BLUE)Preprocessing GD2 tag data...$(RESET)"
	@echo "$(YELLOW)NOTE: This requires raw tag exports in Data/GD2/tag_codes_raw/$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/preprocess_tag_files.py --raw_dir Data/GD2/tag_codes_raw/ --output_dir Data/GD2/tags/

preprocess-tags-gd3:
	@echo "$(BLUE)Preprocessing GD3 tag data...$(RESET)"
	@echo "$(YELLOW)NOTE: This requires raw tag exports in Data/GD3/tag_codes_raw/$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/preprocess_tag_files.py --raw_dir Data/GD3/tag_codes_raw/ --output_dir Data/GD3/tags/

# Full analysis pipeline
analyze-gd1:
	@echo "$(BLUE)Running full analysis pipeline on GD1...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/analyze_dialogues.py --gd_number 1

analyze-gd2:
	@echo "$(BLUE)Running full analysis pipeline on GD2...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/analyze_dialogues.py --gd_number 2

analyze-gd3:
	@echo "$(BLUE)Running full analysis pipeline on GD3...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/analyze_dialogues.py --gd_number 3

analyze-all: analyze-gd1 analyze-gd2 analyze-gd3
	@echo "$(GREEN)All datasets analyzed successfully!$(RESET)"

# Individual analysis commands
consensus-gd1:
	@echo "$(BLUE)Calculating consensus metrics for GD1...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/calculate_consensus.py --gd_number 1

consensus-gd2:
	@echo "$(BLUE)Calculating consensus metrics for GD2...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/calculate_consensus.py --gd_number 2

consensus-gd3:
	@echo "$(BLUE)Calculating consensus metrics for GD3...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/calculate_consensus.py --gd_number 3

divergence-gd1:
	@echo "$(BLUE)Calculating divergence metrics for GD1...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/calculate_divergence.py --gd_number 1

divergence-gd2:
	@echo "$(BLUE)Calculating divergence metrics for GD2...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/calculate_divergence.py --gd_number 2

divergence-gd3:
	@echo "$(BLUE)Calculating divergence metrics for GD3...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/calculate_divergence.py --gd_number 3

indicators-gd1:
	@echo "$(BLUE)Generating indicator heatmaps for GD1...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/calculate_indicators.py --gd_number 1

indicators-gd2:
	@echo "$(BLUE)Generating indicator heatmaps for GD2...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/calculate_indicators.py --gd_number 2

indicators-gd3:
	@echo "$(BLUE)Generating indicator heatmaps for GD3...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/calculate_indicators.py --gd_number 3

tags-gd1:
	@echo "$(BLUE)Analyzing tags for GD1...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/calculate_tags.py --gd_number 1

tags-gd2:
	@echo "$(BLUE)Analyzing tags for GD2...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/calculate_tags.py --gd_number 2

tags-gd3:
	@echo "$(BLUE)Analyzing tags for GD3...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/calculate_tags.py --gd_number 3

# Generic PRI commands using variables
pri:
	@if [ -z "$(GD)" ]; then \
		echo "$(RED)Error: Please specify GD number$(RESET)"; \
		echo "$(YELLOW)Usage: make pri GD=<N>$(RESET)"; \
		echo "$(YELLOW)Example: make pri GD=3$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Calculating participant reliability index for GD$(GD) (traditional metrics)...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/calculate_pri.py --gd_number $(GD)

pri-llm:
	@if [ -z "$(GD)" ]; then \
		echo "$(RED)Error: Please specify GD number$(RESET)"; \
		echo "$(YELLOW)Usage: make pri-llm GD=<N>$(RESET)"; \
		echo "$(YELLOW)Example: make pri-llm GD=3$(RESET)"; \
		echo "$(YELLOW)NOTE: This requires an OpenRouter API key in .env file$(RESET)"; \
		exit 1; \
	fi
	@if [ ! -f .env ]; then \
		echo "$(RED)Error: .env file with OPENROUTER_API_KEY not found$(RESET)"; \
		echo "$(YELLOW)Please create a .env file with your OpenRouter API key$(RESET)"; \
		echo "$(YELLOW)Example: echo 'OPENROUTER_API_KEY=your_key_here' > .env$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Calculating participant reliability index for GD$(GD) with LLM judge...$(RESET)"
	@echo "$(YELLOW)NOTE: This will use OpenRouter API and incur costs$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/calculate_pri.py --gd_number $(GD) --llm-judge

# Specific GD commands (for backwards compatibility)
pri-gd1:
	@echo "$(BLUE)Calculating participant reliability index for GD1...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/calculate_pri.py --gd_number 1

pri-gd2:
	@echo "$(BLUE)Calculating participant reliability index for GD2...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/calculate_pri.py --gd_number 2

pri-gd3:
	@echo "$(BLUE)Calculating participant reliability index for GD3...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/calculate_pri.py --gd_number 3

# Specific GD commands with LLM judge
pri-llm-gd1:
	@if [ ! -f .env ]; then \
		echo "$(RED)Error: .env file with OPENROUTER_API_KEY not found$(RESET)"; \
		echo "$(YELLOW)Please create a .env file with your OpenRouter API key$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Calculating participant reliability index for GD1 with LLM judge...$(RESET)"
	@echo "$(YELLOW)NOTE: This will use OpenRouter API and incur costs$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/calculate_pri.py --gd_number 1 --llm-judge

pri-llm-gd2:
	@if [ ! -f .env ]; then \
		echo "$(RED)Error: .env file with OPENROUTER_API_KEY not found$(RESET)"; \
		echo "$(YELLOW)Please create a .env file with your OpenRouter API key$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Calculating participant reliability index for GD2 with LLM judge...$(RESET)"
	@echo "$(YELLOW)NOTE: This will use OpenRouter API and incur costs$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/calculate_pri.py --gd_number 2 --llm-judge

pri-llm-gd3:
	@if [ ! -f .env ]; then \
		echo "$(RED)Error: .env file with OPENROUTER_API_KEY not found$(RESET)"; \
		echo "$(YELLOW)Please create a .env file with your OpenRouter API key$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Calculating participant reliability index for GD3 with LLM judge...$(RESET)"
	@echo "$(YELLOW)NOTE: This will use OpenRouter API and incur costs$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/calculate_pri.py --gd_number 3 --llm-judge

# Embeddings download
download-embeddings:
	@echo "$(BLUE)Downloading embeddings files...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/download_embeddings.py --list
	@echo ""
	@echo "$(BLUE)To download embeddings for a specific Global Dialogue:$(RESET)"
	@echo "  $(GREEN)make download-embeddings-gd1$(RESET) - Download for GD1"
	@echo "  $(GREEN)make download-embeddings-gd2$(RESET) - Download for GD2"
	@echo "  $(GREEN)make download-embeddings-gd3$(RESET) - Download for GD3"
	@echo "  $(GREEN)make download-all-embeddings$(RESET) - Download all embeddings"

download-embeddings-gd1:
	$(PYTHON) $(TOOLS_DIR)/download_embeddings.py 1

download-embeddings-gd2:
	$(PYTHON) $(TOOLS_DIR)/download_embeddings.py 2

download-embeddings-gd3:
	$(PYTHON) $(TOOLS_DIR)/download_embeddings.py 3

download-all-embeddings:
	$(PYTHON) $(TOOLS_DIR)/download_embeddings.py --all

# Advanced analysis commands
run-thematic-ranking:
	@echo "$(BLUE)Running thematic ranking analysis...$(RESET)"
	@echo "$(YELLOW)NOTE: This requires an OpenAI API key in .env and GD<N>_embeddings.json$(RESET)"
	@if [ ! -f .env ]; then \
		echo "$(RED)Error: .env file with OPENAI_API_KEY not found$(RESET)"; \
		echo "$(YELLOW)Please create a .env file with your OpenAI API key$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Please specify which Global Dialogue to analyze:$(RESET)"
	@echo "  $(GREEN)make run-thematic-ranking-gd1$(RESET) - Run for GD1"
	@echo "  $(GREEN)make run-thematic-ranking-gd2$(RESET) - Run for GD2"
	@echo "  $(GREEN)make run-thematic-ranking-gd3$(RESET) - Run for GD3"

run-thematic-ranking-gd1:
	@echo "$(BLUE)Running thematic ranking analysis for GD1...$(RESET)"
	@if [ ! -f Data/GD1/GD1_embeddings.json ]; then \
		echo "$(RED)Error: Data/GD1/GD1_embeddings.json not found$(RESET)"; \
		echo "$(YELLOW)Please download this file as described in the README$(RESET)"; \
		exit 1; \
	fi
	$(PYTHON) $(ANALYSIS_DIR)/thematic_ranking.py --gd 1

run-thematic-ranking-gd2:
	@echo "$(BLUE)Running thematic ranking analysis for GD2...$(RESET)"
	@if [ ! -f Data/GD2/GD2_embeddings.json ]; then \
		echo "$(RED)Error: Data/GD2/GD2_embeddings.json not found$(RESET)"; \
		echo "$(YELLOW)Please download this file as described in the README$(RESET)"; \
		exit 1; \
	fi
	$(PYTHON) $(ANALYSIS_DIR)/thematic_ranking.py --gd 2

run-thematic-ranking-gd3:
	@echo "$(BLUE)Running thematic ranking analysis for GD3...$(RESET)"
	@if [ ! -f Data/GD3/GD3_embeddings.json ]; then \
		echo "$(RED)Error: Data/GD3/GD3_embeddings.json not found$(RESET)"; \
		echo "$(YELLOW)Please download this file as described in the README$(RESET)"; \
		exit 1; \
	fi
	$(PYTHON) $(ANALYSIS_DIR)/thematic_ranking.py --gd 3

# Preview CSV files
preview-csvs-gd1:
	@echo "$(BLUE)Previewing CSV files in GD1 directory...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/preview_csvs.py --gd_number 1

preview-csvs-gd2:
	@echo "$(BLUE)Previewing CSV files in GD2 directory...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/preview_csvs.py --gd_number 2

preview-csvs-gd3:
	@echo "$(BLUE)Previewing CSV files in GD3 directory...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/preview_csvs.py --gd_number 3

# Utilities
clean:
	@echo "$(BLUE)Cleaning up cache and temporary files...$(RESET)"
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -delete
	@find . -name "processed_data.pkl" -delete
	@find . -name "*.bak" -delete
	@echo "$(GREEN)Cleanup complete!$(RESET)"
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
        preprocess-all preprocess-tags analyze-all \
        consensus divergence indicators tags \
        download-embeddings download-all-embeddings \
        run-thematic-ranking \
        pri pri-llm export-unreliable \
        preview-csvs

# Default target
help:
	@echo "$(BLUE)Global Dialogues Analysis Commands:$(RESET)"
	@echo "  $(GREEN)make help$(RESET)                 - Show this help message"
	@echo ""
	@echo "$(BLUE)Preprocessing Commands:$(RESET)"
	@echo "  $(GREEN)make preprocess GD=<N>$(RESET)    - Preprocess GD<N> data (metadata cleanup + standardize aggregate)"
	@echo "  $(GREEN)make preprocess-tags GD=<N>$(RESET) - Preprocess tag data for GD<N>"
	@echo "  $(GREEN)make preprocess-all$(RESET)       - Preprocess all GD datasets"
	@echo ""
	@echo "$(BLUE)Data Commands:$(RESET)"
	@echo "  $(GREEN)make download-embeddings GD=<N>$(RESET) - Download embeddings for GD<N>"
	@echo "  $(GREEN)make download-embeddings$(RESET)  - Show available embedding files and download options"
	@echo "  $(GREEN)make download-all-embeddings$(RESET) - Download all embeddings"
	@echo ""
	@echo "$(BLUE)Analysis Commands:$(RESET)"
	@echo "  $(GREEN)make analyze GD=<N>$(RESET)       - Run full analysis pipeline on GD<N>"
	@echo "  $(GREEN)make analyze-all$(RESET)          - Run full analysis pipeline on all GD datasets"
	@echo ""
	@echo "$(BLUE)Individual Analysis Commands:$(RESET)"
	@echo "  $(GREEN)make consensus GD=<N>$(RESET)     - Calculate consensus metrics for GD<N>"
	@echo "  $(GREEN)make divergence GD=<N>$(RESET)    - Calculate divergence metrics for GD<N>"
	@echo "  $(GREEN)make indicators GD=<N>$(RESET)    - Generate indicator heatmaps for GD<N>"
	@echo "  $(GREEN)make tags GD=<N>$(RESET)          - Analyze tags for GD<N>"
	@echo ""
	@echo "$(BLUE)PRI (Participant Reliability Index) Commands:$(RESET)"
	@echo "  $(GREEN)make pri GD=<N>$(RESET)           - Calculate PRI for GD<N> (traditional metrics only)"
	@echo "  $(GREEN)make pri-llm GD=<N>$(RESET)       - Calculate PRI for GD<N> with LLM judge assessment"
	@echo "  $(GREEN)make pri-gd<N>$(RESET)            - Calculate PRI for specific GD (traditional metrics only)"
	@echo "  $(GREEN)make pri-llm-gd<N>$(RESET)        - Calculate PRI for specific GD with LLM judge"
	@echo "  $(GREEN)make export-unreliable GD=<N>$(RESET) - Export unreliable participants CSV for GD<N>"
	@echo "  $(GREEN)make export-unreliable-gd<N>$(RESET) - Export unreliable participants CSV for specific GD"
	@echo ""
	@echo "$(BLUE)Advanced Analysis Commands:$(RESET)"
	@echo "  $(GREEN)make run-thematic-ranking GD=<N>$(RESET) - Run thematic ranking for GD<N> (requires API key and embeddings)"
	@echo "  $(GREEN)make run-thematic-ranking$(RESET) - Show thematic ranking options"
	@echo ""
	@echo "$(BLUE)Utilities:$(RESET)"
	@echo "  $(GREEN)make preview-csvs GD=<N>$(RESET)  - Preview all CSV files in GD<N> directory"
	@echo "  $(GREEN)make clean$(RESET)                - Clean up cache and temporary files"

# Preprocessing commands using variables
preprocess:
	@if [ -z "$(GD)" ]; then \
		echo "$(RED)Error: Please specify GD number$(RESET)"; \
		echo "$(YELLOW)Usage: make preprocess GD=<N>$(RESET)"; \
		echo "$(YELLOW)Example: make preprocess GD=3$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Preprocessing GD$(GD) data...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/preprocess_cleanup_metadata.py --gd_number $(GD)
	$(PYTHON) $(TOOLS_DIR)/preprocess_aggregate.py --gd_number $(GD)

preprocess-all:
	@echo "$(BLUE)Preprocessing all GD datasets...$(RESET)"
	@for gd in 1 2 3; do \
		echo "$(BLUE)Preprocessing GD$$gd data...$(RESET)"; \
		$(PYTHON) $(TOOLS_DIR)/preprocess_cleanup_metadata.py --gd_number $$gd; \
		$(PYTHON) $(TOOLS_DIR)/preprocess_aggregate.py --gd_number $$gd; \
	done
	@echo "$(GREEN)All datasets preprocessed successfully!$(RESET)"

# Tag preprocessing using variables
preprocess-tags:
	@if [ -z "$(GD)" ]; then \
		echo "$(RED)Error: Please specify GD number$(RESET)"; \
		echo "$(YELLOW)Usage: make preprocess-tags GD=<N>$(RESET)"; \
		echo "$(YELLOW)Example: make preprocess-tags GD=3$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Preprocessing GD$(GD) tag data...$(RESET)"
	@echo "$(YELLOW)NOTE: This requires raw tag exports in Data/GD$(GD)/tag_codes_raw/$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/preprocess_tag_files.py --raw_dir Data/GD$(GD)/tag_codes_raw/ --output_dir Data/GD$(GD)/tags/

# Analysis pipeline using variables
analyze:
	@if [ -z "$(GD)" ]; then \
		echo "$(RED)Error: Please specify GD number$(RESET)"; \
		echo "$(YELLOW)Usage: make analyze GD=<N>$(RESET)"; \
		echo "$(YELLOW)Example: make analyze GD=3$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Running full analysis pipeline on GD$(GD)...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/analyze_dialogues.py --gd_number $(GD)

analyze-all:
	@echo "$(BLUE)Running full analysis pipeline on all GD datasets...$(RESET)"
	@for gd in 1 2 3; do \
		echo "$(BLUE)Running full analysis pipeline on GD$$gd...$(RESET)"; \
		$(PYTHON) $(TOOLS_DIR)/analyze_dialogues.py --gd_number $$gd; \
	done
	@echo "$(GREEN)All datasets analyzed successfully!$(RESET)"

# Individual analysis commands using variables
consensus:
	@if [ -z "$(GD)" ]; then \
		echo "$(RED)Error: Please specify GD number$(RESET)"; \
		echo "$(YELLOW)Usage: make consensus GD=<N>$(RESET)"; \
		echo "$(YELLOW)Example: make consensus GD=3$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Calculating consensus metrics for GD$(GD)...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/calculate_consensus.py --gd_number $(GD)

divergence:
	@if [ -z "$(GD)" ]; then \
		echo "$(RED)Error: Please specify GD number$(RESET)"; \
		echo "$(YELLOW)Usage: make divergence GD=<N>$(RESET)"; \
		echo "$(YELLOW)Example: make divergence GD=3$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Calculating divergence metrics for GD$(GD)...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/calculate_divergence.py --gd_number $(GD)

indicators:
	@if [ -z "$(GD)" ]; then \
		echo "$(RED)Error: Please specify GD number$(RESET)"; \
		echo "$(YELLOW)Usage: make indicators GD=<N>$(RESET)"; \
		echo "$(YELLOW)Example: make indicators GD=3$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Generating indicator heatmaps for GD$(GD)...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/calculate_indicators.py --gd_number $(GD)

tags:
	@if [ -z "$(GD)" ]; then \
		echo "$(RED)Error: Please specify GD number$(RESET)"; \
		echo "$(YELLOW)Usage: make tags GD=<N>$(RESET)"; \
		echo "$(YELLOW)Example: make tags GD=3$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Analyzing tags for GD$(GD)...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/calculate_tags.py --gd_number $(GD)

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

# Export unreliable participants
export-unreliable:
	@if [ -z "$(GD)" ]; then \
		echo "$(RED)Error: Please specify GD number$(RESET)"; \
		echo "$(YELLOW)Usage: make export-unreliable GD=<N>$(RESET)"; \
		echo "$(YELLOW)Example: make export-unreliable GD=3$(RESET)"; \
		echo "$(YELLOW)Optional: METHOD=outliers|percentile|threshold THRESHOLD=<value>$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Exporting unreliable participants for GD$(GD)...$(RESET)"
	@if [ -n "$(METHOD)" ] && [ -n "$(THRESHOLD)" ]; then \
		$(PYTHON) $(TOOLS_DIR)/export_unreliable_participants.py $(GD) --method $(METHOD) --threshold $(THRESHOLD); \
	elif [ -n "$(METHOD)" ]; then \
		$(PYTHON) $(TOOLS_DIR)/export_unreliable_participants.py $(GD) --method $(METHOD); \
	else \
		$(PYTHON) $(TOOLS_DIR)/export_unreliable_participants.py $(GD); \
	fi

# Embeddings download using variables
download-embeddings:
	@if [ -z "$(GD)" ]; then \
		echo "$(BLUE)Downloading embeddings files...$(RESET)"; \
		$(PYTHON) $(TOOLS_DIR)/download_embeddings.py --list; \
		echo ""; \
		echo "$(BLUE)To download embeddings for a specific Global Dialogue:$(RESET)"; \
		echo "  $(GREEN)make download-embeddings GD=<N>$(RESET) - Download for GD<N>"; \
		echo "  $(GREEN)make download-all-embeddings$(RESET) - Download all embeddings"; \
	else \
		echo "$(BLUE)Downloading embeddings for GD$(GD)...$(RESET)"; \
		$(PYTHON) $(TOOLS_DIR)/download_embeddings.py $(GD); \
	fi

download-all-embeddings:
	$(PYTHON) $(TOOLS_DIR)/download_embeddings.py --all

# Thematic ranking using variables
run-thematic-ranking:
	@if [ -z "$(GD)" ]; then \
		echo "$(BLUE)Running thematic ranking analysis...$(RESET)"; \
		echo "$(YELLOW)NOTE: This requires an OpenAI API key in .env and GD<N>_embeddings.json$(RESET)"; \
		echo "$(BLUE)Please specify which Global Dialogue to analyze:$(RESET)"; \
		echo "  $(GREEN)make run-thematic-ranking GD=<N>$(RESET) - Run for GD<N>"; \
		exit 1; \
	fi
	@if [ ! -f .env ]; then \
		echo "$(RED)Error: .env file with OPENAI_API_KEY not found$(RESET)"; \
		echo "$(YELLOW)Please create a .env file with your OpenAI API key$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Running thematic ranking analysis for GD$(GD)...$(RESET)"
	@if [ ! -f Data/GD$(GD)/GD$(GD)_embeddings.json ]; then \
		echo "$(RED)Error: Data/GD$(GD)/GD$(GD)_embeddings.json not found$(RESET)"; \
		echo "$(YELLOW)Please download this file as described in the README$(RESET)"; \
		exit 1; \
	fi
	$(PYTHON) $(ANALYSIS_DIR)/thematic_ranking.py --gd $(GD)

# CSV preview using variables
preview-csvs:
	@if [ -z "$(GD)" ]; then \
		echo "$(RED)Error: Please specify GD number$(RESET)"; \
		echo "$(YELLOW)Usage: make preview-csvs GD=<N>$(RESET)"; \
		echo "$(YELLOW)Example: make preview-csvs GD=3$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Previewing CSV files in GD$(GD) directory...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/preview_csvs.py --gd_number $(GD)

# Utilities
clean:
	@echo "$(BLUE)Cleaning up cache and temporary files...$(RESET)"
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -delete
	@find . -name "processed_data.pkl" -delete
	@find . -name "*.bak" -delete
	@echo "$(GREEN)Cleanup complete!$(RESET)"
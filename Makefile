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
        run-thematic-ranking

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
	@echo "  $(GREEN)make pri-gd<N>$(RESET)            - Calculate participant reliability index for GD<N>"
	@echo ""
	@echo "$(BLUE)Advanced Analysis Commands:$(RESET)"
	@echo "  $(GREEN)make run-thematic-ranking$(RESET) - Run thematic ranking analysis (requires API key and embeddings)"
	@echo ""
	@echo "$(BLUE)Utilities:$(RESET)"
	@echo "  $(GREEN)make preview-csvs-gd<N>$(RESET)   - Preview all CSV files in GD<N> directory"
	@echo "  $(GREEN)make clean$(RESET)                - Clean up cache and temporary files"

# Preprocessing commands (metadata cleanup + standardize aggregate)
preprocess-gd1:
	@echo "$(BLUE)Preprocessing GD1 data...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/preprocess_cleanup_metadata.py 1
	$(PYTHON) $(TOOLS_DIR)/preprocess_aggregate.py --gd_number 1

preprocess-gd2:
	@echo "$(BLUE)Preprocessing GD2 data...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/preprocess_cleanup_metadata.py 2
	$(PYTHON) $(TOOLS_DIR)/preprocess_aggregate.py --gd_number 2

preprocess-gd3:
	@echo "$(BLUE)Preprocessing GD3 data...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/preprocess_cleanup_metadata.py 3
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
	$(PYTHON) $(TOOLS_DIR)/analyze_dialogues.py 1

analyze-gd2:
	@echo "$(BLUE)Running full analysis pipeline on GD2...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/analyze_dialogues.py 2

analyze-gd3:
	@echo "$(BLUE)Running full analysis pipeline on GD3...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/analyze_dialogues.py 3

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
	$(PYTHON) $(TOOLS_DIR)/calculate_tags.py 1

tags-gd2:
	@echo "$(BLUE)Analyzing tags for GD2...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/calculate_tags.py 2

tags-gd3:
	@echo "$(BLUE)Analyzing tags for GD3...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/calculate_tags.py 3

pri-gd1:
	@echo "$(BLUE)Calculating participant reliability index for GD1...$(RESET)"
	@echo "$(YELLOW)NOTE: Making temporary edits to calculate_pri.py to use GD1 data$(RESET)"
	sed -i'.bak' 's/GD_NUMBER = 3/GD_NUMBER = 1/' $(TOOLS_DIR)/calculate_pri.py
	$(PYTHON) $(TOOLS_DIR)/calculate_pri.py
	mv $(TOOLS_DIR)/calculate_pri.py.bak $(TOOLS_DIR)/calculate_pri.py

pri-gd2:
	@echo "$(BLUE)Calculating participant reliability index for GD2...$(RESET)"
	@echo "$(YELLOW)NOTE: Making temporary edits to calculate_pri.py to use GD2 data$(RESET)"
	sed -i'.bak' 's/GD_NUMBER = 3/GD_NUMBER = 2/' $(TOOLS_DIR)/calculate_pri.py
	$(PYTHON) $(TOOLS_DIR)/calculate_pri.py
	mv $(TOOLS_DIR)/calculate_pri.py.bak $(TOOLS_DIR)/calculate_pri.py

pri-gd3:
	@echo "$(BLUE)Calculating participant reliability index for GD3...$(RESET)"
	$(PYTHON) $(TOOLS_DIR)/calculate_pri.py

# Advanced analysis commands
run-thematic-ranking:
	@echo "$(BLUE)Running thematic ranking analysis...$(RESET)"
	@echo "$(YELLOW)NOTE: This requires an OpenAI API key in .env and GD3_embeddings.json$(RESET)"
	@if [ ! -f Data/GD3/GD3_embeddings.json ]; then \
		echo "$(RED)Error: Data/GD3/GD3_embeddings.json not found$(RESET)"; \
		echo "$(YELLOW)Please download this file as described in the README$(RESET)"; \
		exit 1; \
	fi
	@if [ ! -f .env ]; then \
		echo "$(RED)Error: .env file with OPENAI_API_KEY not found$(RESET)"; \
		echo "$(YELLOW)Please create a .env file with your OpenAI API key$(RESET)"; \
		exit 1; \
	fi
	$(PYTHON) $(ANALYSIS_DIR)/thematic_ranking.py

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
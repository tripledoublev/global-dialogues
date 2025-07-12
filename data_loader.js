/**
 * Comprehensive Data Loader for Global Dialogues Survey Explorer
 * Handles loading CSV data from all GD rounds with error handling and fallbacks
 */

class GlobalDialoguesDataLoader {
    constructor() {
        this.rounds = ['GD1', 'GD2', 'GD3', 'GD4'];
        this.dataTypes = {
            responses: 'verbatim_map',
            tags: 'tags/all_thought_labels',
            tagAnalysis: 'tags/tag_analysis_report',
            consensus: 'consensus/consensus_profiles',
            consensusBySegment: 'consensus/major_segment_min_agreement_top10',
            divergence: 'divergence/divergence_by_question',
            participants: 'aggregate_standardized',
            binary: 'binary',
            preference: 'preference',
            pri: 'pri/pri_scores'
        };
        this.loadedData = {};
        this.loadingStatus = {};
    }

    /**
     * Load all available data for specified rounds
     */
    async loadAllData(rounds = this.rounds) {
        const loadingPromises = [];
        
        for (const round of rounds) {
            this.loadingStatus[round] = {
                total: Object.keys(this.dataTypes).length,
                loaded: 0,
                errors: []
            };

            // Load each data type for this round
            for (const [dataType, fileName] of Object.entries(this.dataTypes)) {
                const promise = this.loadDataType(round, dataType, fileName)
                    .then(() => {
                        this.loadingStatus[round].loaded++;
                    })
                    .catch(error => {
                        this.loadingStatus[round].errors.push({
                            dataType,
                            error: error.message
                        });
                        console.warn(`Failed to load ${dataType} for ${round}:`, error);
                    });
                
                loadingPromises.push(promise);
            }
        }

        await Promise.allSettled(loadingPromises);
        return this.processLoadedData();
    }

    /**
     * Load a specific data type for a round
     */
    async loadDataType(round, dataType, fileName) {
        const paths = this.getDataPaths(round, dataType, fileName);
        
        for (const path of paths) {
            try {
                const data = await this.loadCSV(path);
                
                // Initialize round data if not exists
                if (!this.loadedData[round]) {
                    this.loadedData[round] = {};
                }
                
                this.loadedData[round][dataType] = data;
                console.log(`✅ Loaded ${dataType} for ${round}: ${data.length} records`);
                return data;
                
            } catch (error) {
                console.warn(`Failed to load ${path}:`, error);
                continue;
            }
        }
        
        throw new Error(`Could not load ${dataType} for ${round} from any path`);
    }

    /**
     * Get possible file paths for a data type
     */
    getDataPaths(round, dataType, fileName) {
        const paths = [];
        
        // Analysis output paths
        if (['consensus', 'divergence', 'tagAnalysis', 'pri', 'consensusBySegment'].includes(dataType)) {
            paths.push(`analysis_output/${round}/${fileName}.csv`);
        }
        
        // Data directory paths
        if (['responses', 'tags', 'participants', 'binary', 'preference'].includes(dataType)) {
            if (dataType === 'responses') {
                paths.push(`Data/${round}/${round}_${fileName}.csv`);
            } else if (dataType === 'tags') {
                paths.push(`Data/${round}/${fileName}.csv`);
            } else if (dataType === 'participants') {
                paths.push(`Data/${round}/${round}_${fileName}.csv`);
            } else if (dataType === 'binary') {
                paths.push(`Data/${round}/${round}_${fileName}.csv`);
            } else if (dataType === 'preference') {
                paths.push(`Data/${round}/${round}_${fileName}.csv`);
            }
        }
        
        return paths;
    }

    /**
     * Load CSV file using Papa Parse
     */
    async loadCSV(url) {
        return new Promise((resolve, reject) => {
            Papa.parse(url, {
                download: true,
                header: true,
                skipEmptyLines: true,
                complete: (results) => {
                    if (results.errors.length > 0) {
                        const criticalErrors = results.errors.filter(error => 
                            error.type === 'Delimiter' || error.type === 'Quotes'
                        );
                        
                        if (criticalErrors.length > 0) {
                            reject(new Error(`CSV parsing errors: ${criticalErrors.map(e => e.message).join(', ')}`));
                            return;
                        }
                    }
                    
                    // Filter out empty rows
                    const cleanData = results.data.filter(row => 
                        row && Object.values(row).some(val => val && val.toString().trim())
                    );
                    
                    resolve(cleanData);
                },
                error: (error) => {
                    reject(new Error(`Failed to fetch ${url}: ${error.message}`));
                }
            });
        });
    }

    /**
     * Process and merge loaded data
     */
    processLoadedData() {
        const processedData = {
            responses: [],
            consensus: [],
            divergence: [],
            tags: [],
            tagAnalysis: [],
            participants: [],
            binary: [],
            pri: [],
            preference: [],
            consensusBySegment: [],
            metadata: {
                rounds: {},
                totals: {},
                loadingStatus: this.loadingStatus
            }
        };

        // Combine data from all rounds
        for (const [round, roundData] of Object.entries(this.loadedData)) {
            processedData.metadata.rounds[round] = {
                dataTypes: Object.keys(roundData),
                counts: {}
            };

            for (const [dataType, data] of Object.entries(roundData)) {
                if (data && Array.isArray(data)) {
                    // Add round information to each record
                    const dataWithRound = data.map(record => ({
                        ...record,
                        _round: round,
                        _dataType: dataType
                    }));

                    processedData[dataType].push(...dataWithRound);
                    processedData.metadata.rounds[round].counts[dataType] = data.length;
                }
            }
        }

        // Calculate totals
        for (const dataType of Object.keys(processedData)) {
            if (Array.isArray(processedData[dataType])) {
                processedData.metadata.totals[dataType] = processedData[dataType].length;
            }
        }

        // Create a map of tags to categories for efficient lookup
        const tagToCategoryMap = new Map();
        if (processedData.tagAnalysis) {
            processedData.tagAnalysis.forEach(item => {
                if (item.Tag && item.Category) {
                    tagToCategoryMap.set(item.Tag.trim(), item.Category.trim());
                }
            });
        }

        // Create a map of PRI scores
        const priScoreMap = new Map();
        if(processedData.pri) {
            processedData.pri.forEach(item => {
                if(item['Participant ID'] && item['PRI_Score']) {
                    priScoreMap.set(item['Participant ID'], parseFloat(item['PRI_Score']));
                }
            });
        }

        // Process response-tag relationships and add categories
        this.processResponseTags(processedData, tagToCategoryMap);

        // Add PRI scores to responses and participants
        processedData.responses.forEach(response => {
            response.PRI_Score = priScoreMap.get(response['Participant ID']) || null;
        });
        processedData.participants.forEach(participant => {
            participant.PRI_Score = priScoreMap.get(participant['Participant ID']) || null;
        });

        console.log('✅ Data processing complete:', processedData.metadata.totals);
        return processedData;
    }

    /**
     * Merge response data with tag data and category data
     */
    processResponseTags(processedData, tagToCategoryMap) {
        const tagMap = new Map();
        
        // Create a map of tags by question and participant from 'all_thought_labels'
        processedData.tags.forEach(tagRecord => {
            const key = `${tagRecord['Question ID']}-${tagRecord['Participant ID']}`;
            tagMap.set(key, tagRecord);
        });

        // Add tag and category information to each response
        processedData.responses.forEach(response => {
            const key = `${response['Question ID']}-${response['Participant ID']}`;
            const tagData = tagMap.get(key);
            
            response.tags = [];
            response.categories = [];
            
            if (tagData) {
                const categories = new Set();
                response.sentiment = tagData.Sentiment;
                
                // Extract all non-empty tags for the response
                for (let i = 1; i <= 10; i++) {
                    const tag = tagData[`Tag ${i}`];
                    if (tag && tag.toString().trim()) {
                        const tagName = tag.toString().trim();
                        response.tags.push(tagName);
                        
                        // Use the map to find the category for the tag
                        if (tagToCategoryMap.has(tagName)) {
                            categories.add(tagToCategoryMap.get(tagName));
                        }
                    }
                }
                response.categories = Array.from(categories);
            }
        });
    }

    /**
     * Get loading progress for UI updates
     */
    getLoadingProgress() {
        const progress = {};
        
        for (const [round, status] of Object.entries(this.loadingStatus)) {
            progress[round] = {
                percentage: Math.round((status.loaded / status.total) * 100),
                loaded: status.loaded,
                total: status.total,
                errors: status.errors.length
            };
        }
        
        return progress;
    }

    /**
     * Get unique values for filter building
     */
    getUniqueValues(data, field) {
        const values = new Set();
        
        data.forEach(record => {
            const value = record[field];
            if (value && value.toString().trim()) {
                values.add(value.toString().trim());
            }
        });
        
        return Array.from(values).sort();
    }

    /**
     * Build filter options from loaded data
     */
    buildFilterOptions(processedData) {
        const filters = {
            rounds: this.getUniqueValues(processedData.responses, '_round'),
            questions: this.getUniqueValues(processedData.responses, 'Question ID'),
            participants: this.getUniqueValues(processedData.responses, 'Participant ID'),
            categories: this.getUniqueValues(processedData.tagAnalysis, 'Category'),
            tags: [],
            sentiments: this.getUniqueValues(processedData.responses, 'sentiment'),
            countries: [],
            languages: []
        };

        // Extract all tags from responses
        const allTags = new Set();
        processedData.responses.forEach(response => {
            if (response.tags && Array.isArray(response.tags)) {
                response.tags.forEach(tag => allTags.add(tag));
            }
        });
        filters.tags = Array.from(allTags).sort();

        const participantData = processedData.participants;
        const countries = new Set();
        const languages = new Set();

        participantData.forEach(p => {
            if(p.Language) languages.add(p.Language);
            for(const key in p) {
                if(key.startsWith('O7:')) {
                    countries.add(key.replace('O7: ', ''));
                }
            }
        });

        filters.countries = Array.from(countries).sort();
        filters.languages = Array.from(languages).sort();

        return filters;
    }

    /**
     * Export data for external use
     */
    exportData(processedData, format = 'json') {
        if (format === 'json') {
            return JSON.stringify(processedData, null, 2);
        } else if (format === 'csv') {
            // Export as CSV (simplified)
            const csvData = processedData.responses.map(response => ({
                Round: response._round,
                QuestionID: response['Question ID'],
                QuestionText: response['Question Text'],
                ParticipantID: response['Participant ID'],
                ResponseText: response['Thought Text'] || response['Response Text'],
                Tags: response.tags ? response.tags.join('; ') : '',
                Sentiment: response.sentiment || '',
                PRI_Score: response.PRI_Score || ''
            }));
            
            return Papa.unparse(csvData);
        }
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GlobalDialoguesDataLoader;
} else if (typeof window !== 'undefined') {
    window.GlobalDialoguesDataLoader = GlobalDialoguesDataLoader;
} 
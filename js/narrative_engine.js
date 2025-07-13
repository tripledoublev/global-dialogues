class NarrativeEngine {
    constructor() {
        this.themes = {};
        this.currentNarrative = null;
        this.typewriter = null;
        this.isPlaying = false;
        // Don't auto-initialize - wait for user to click start
    }
    
    // Method to stop current processes
    stopCurrentProcesses() {
        if (this.typewriter) {
            this.typewriter.pause();
            this.typewriter = null;
        }
        this.isPlaying = false;
        
        // Clear any existing choices
        const viewport = document.getElementById('story-viewport');
        if (viewport) {
            const existingChoices = viewport.querySelector('.branching-choices');
            if (existingChoices) existingChoices.remove();
        }
    }
    
    // Reset narrative state for new content
    resetNarrativeState() {
        this.currentNarrative = null;
        this.isPlaying = false;
        
        const viewport = document.getElementById('story-viewport');
        if (viewport) {
            viewport.classList.remove('active');
            viewport.classList.add('active'); // Re-add to trigger animation
        }
    }
    
    // Clean up markdown from text - more conservative approach
    cleanMarkdown(text) {
        if (!text) return '';
        return text
            .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold markdown only
            .replace(/^\s*[-*+]\s*/gm, '') // Remove list markers at start of lines
            .replace(/^\s*\d+\.\s*/gm, '') // Remove numbered lists at start of lines
            .trim();
    }
    
    // Process and clean theme data
    processThemeData(themes) {
        const processedThemes = {};
        
        for (const [themeName, theme] of Object.entries(themes)) {
            const cleanThemeName = this.cleanMarkdown(themeName);
            processedThemes[cleanThemeName] = {
                ...theme,
                subthemes: theme.subthemes.map(subtheme => ({
                    ...subtheme,
                    name: this.cleanMarkdown(subtheme.name),
                    question: this.cleanMarkdown(subtheme.question),
                    response_reactions: subtheme.response_reactions ? subtheme.response_reactions.map(rr => ({
                        response: this.cleanMarkdown(rr.response),
                        reaction: this.cleanMarkdown(rr.reaction)
                    })) : []
                }))
            };
        }
        
        return processedThemes;
    }
    
    async init() {
        await this.loadThemes();
        document.body.classList.add('fade-background');
        this.renderThemeSelector();
        this.setupEventListeners();
        const params = new URLSearchParams(window.location.search);
        const startTheme = params.get('theme') || Object.keys(this.themes)[Math.floor(Math.random() * Object.keys(this.themes).length)];
        if (startTheme && this.themes[startTheme]) {
            this.playNarrative(startTheme);
        }
    }
    
    async loadThemes() {
        try {
            const response = await fetch('./data/compiled_themes.json');
            const rawThemes = await response.json();
            this.themes = this.processThemeData(rawThemes);
            this.renderThemeSelector();
        } catch (error) {
            console.error('Error loading themes:', error);
        }
    }
    
    renderThemeSelector() {
        const container = document.getElementById('theme-constellation');
        if (!container) return;
        
        container.innerHTML = '';
        const sortedThemes = Object.entries(this.themes).sort((a,b) => a[0].localeCompare(b[0])).slice(0,5);
        sortedThemes.forEach(([themeName, theme]) => {
            const themeEl = document.createElement('div');
            themeEl.className = 'theme-tag';
            themeEl.textContent = themeName;
            // Apply theme colors to CSS variables for clean styling
            themeEl.style.setProperty('--theme-primary', theme.visual_style.primary_color);
            themeEl.style.setProperty('--theme-accent', theme.visual_style.accent_color);
            themeEl.setAttribute('role', 'button');
            themeEl.setAttribute('tabindex', '0');
            themeEl.setAttribute('aria-label', `Explore theme: ${themeName}`);
            themeEl.addEventListener('click', () => this.playNarrative(themeName));
            themeEl.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    this.playNarrative(themeName);
                }
            });
            container.appendChild(themeEl);
        });
        
        // Also populate the themes sidebar
        this.populateThemesSidebar();
    }
    
    populateThemesSidebar() {
        const themesList = document.getElementById('themes-list');
        if (!themesList) return;
        
        themesList.innerHTML = '';
        const sortedThemes = Object.entries(this.themes).sort((a,b) => a[0].localeCompare(b[0]));
        
        sortedThemes.forEach(([themeName, theme]) => {
            const themeItem = document.createElement('div');
            themeItem.className = 'theme-item';
            themeItem.textContent = themeName;
            themeItem.setAttribute('role', 'button');
            themeItem.setAttribute('tabindex', '0');
            themeItem.setAttribute('aria-label', `Select theme: ${themeName}`);
            
            themeItem.addEventListener('click', () => {
                this.selectTheme(themeName);
                this.toggleSidebar('themes-sidebar'); // Close the themes sidebar
            });
            
            themeItem.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    this.selectTheme(themeName);
                    this.toggleSidebar('themes-sidebar'); // Close the themes sidebar
                }
            });
            
            themesList.appendChild(themeItem);
        });
    }
    
    populateSubthemesSidebar(themeName) {
        const subthemesList = document.getElementById('subthemes-list');
        if (!subthemesList) return;
        
        subthemesList.innerHTML = '';
        const theme = this.themes[themeName];
        if (!theme || !theme.subthemes) return;
        
        // Clear active states from themes
        document.querySelectorAll('.theme-item').forEach(item => item.classList.remove('active'));
        
        // Set active theme
        const activeThemeItem = Array.from(document.querySelectorAll('.theme-item')).find(item => item.textContent === themeName);
        if (activeThemeItem) {
            activeThemeItem.classList.add('active');
        }
        
        theme.subthemes.forEach(subtheme => {
            const subthemeItem = document.createElement('div');
            subthemeItem.className = 'subtheme-item';
            subthemeItem.textContent = subtheme.name;
            subthemeItem.setAttribute('role', 'button');
            subthemeItem.setAttribute('tabindex', '0');
            subthemeItem.setAttribute('aria-label', `Select subtheme: ${subtheme.name}`);
            
            subthemeItem.addEventListener('click', () => {
                this.selectSubtheme(themeName, subtheme.name);
            });
            
            subthemeItem.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    this.selectSubtheme(themeName, subtheme.name);
                }
            });
            
            subthemesList.appendChild(subthemeItem);
        });
    }
    
    async selectTheme(themeName) {
        // Stop current processes and reset state
        this.stopCurrentProcesses();
        this.resetNarrativeState();
        
        // Show loading state
        const activeThemeItem = Array.from(document.querySelectorAll('.theme-item')).find(item => item.textContent === themeName);
        if (activeThemeItem) {
            this.showLoadingState(activeThemeItem);
        }
        
        // Clear active states from subthemes
        document.querySelectorAll('.subtheme-item').forEach(item => item.classList.remove('active'));
        
        // Clear active states from themes
        document.querySelectorAll('.theme-item').forEach(item => item.classList.remove('active'));
        
        // Set active theme
        if (activeThemeItem) {
            activeThemeItem.classList.add('active');
        }
        
        // Play the theme narrative
        await this.playNarrative(themeName);
        
        // Hide loading state
        if (activeThemeItem) {
            this.hideLoadingState(activeThemeItem);
        }
    }
    
    async selectSubtheme(themeName, subthemeName) {
        // Stop current processes and reset state
        this.stopCurrentProcesses();
        this.resetNarrativeState();
        
        // Show loading state
        const activeSubthemeItem = Array.from(document.querySelectorAll('.subtheme-item')).find(item => item.textContent === subthemeName);
        if (activeSubthemeItem) {
            this.showLoadingState(activeSubthemeItem);
        }
        
        // Clear active states from subthemes
        document.querySelectorAll('.subtheme-item').forEach(item => item.classList.remove('active'));
        
        // Set active subtheme
        if (activeSubthemeItem) {
            activeSubthemeItem.classList.add('active');
        }
        
        // Play the subtheme narrative
        await this.handleChoice(themeName, subthemeName);
        
        // Hide loading state
        if (activeSubthemeItem) {
            this.hideLoadingState(activeSubthemeItem);
        }
    }
    
    async playNarrative(themeName) {
        // Stop current processes
        this.stopCurrentProcesses();
        
        document.getElementById('theme-constellation').style.display = 'none';
        this.currentNarrative = await this.compileNarrative(themeName);
        const viewport = document.getElementById('story-viewport');
        
        // Clear existing content more carefully
        const existingChoices = viewport.querySelector('.branching-choices');
        if (existingChoices) existingChoices.remove();
        
        viewport.classList.add('active');
        let textContent = viewport.querySelector('#text-content');
        if (!textContent) {
            textContent = document.createElement('div');
            textContent.id = 'text-content';
            viewport.appendChild(textContent);
        } else {
            textContent.innerHTML = '';
        }
        
        this.applyThemeStyle(this.currentNarrative.visual_style);
        atmosphere.setMood(this.currentNarrative.visual_style.mood);
        this.typewriter = new ArtisticTypewriter(textContent, { speed: 30, questionSpeed: 15, reactionSpeed: 45, pauseAfterSentence: 500, pauseAfterParagraph: 1000 });
        this.typewriter.play();
        this.isPlaying = true;
        await this.playNextSnippet();
        
        // Populate subthemes sidebar for the selected theme
        this.populateSubthemesSidebar(themeName);
    }

    async playNextSnippet() {
        const narrative = this.currentNarrative;
        if (!narrative.currentSegmentIndex) narrative.currentSegmentIndex = 0;
        const segment = narrative.segments[narrative.currentSegmentIndex];
        if (!segment) {
            this.showBranchingChoices();
            return;
        }
        const textContent = document.getElementById('text-content');
        textContent.classList.remove('fade-in');
        await this.typewriter.typeText(segment.text);
        await this.typewriter.sleep(segment.pause_after);
        narrative.currentSegmentIndex++;
        await this.playNextSnippet();
    }

    showBranchingChoices() {
        const viewport = document.getElementById('story-viewport');
        const choicesDiv = document.createElement('div');
        choicesDiv.className = 'branching-choices';
        
        // Randomize number of choices between 1-3
        const numChoices = Math.floor(Math.random() * 3) + 1;
        
        const otherThemes = Object.keys(this.themes).filter(t => t !== this.currentNarrative.themeName).sort(() => Math.random() - 0.5).slice(0, numChoices);
        const subChoices = [];
        otherThemes.forEach(themeName => {
            const sub = this.themes[themeName].subthemes[Math.floor(Math.random() * this.themes[themeName].subthemes.length)];
            subChoices.push({theme: themeName, subName: sub.name});
        });
        
        viewport.appendChild(choicesDiv);
        
        // Show buttons one after another with delay
        subChoices.forEach((choice, index) => {
            setTimeout(() => {
                const btn = document.createElement('div');
                btn.className = 'theme-tag choice-btn';
                btn.textContent = choice.subName;
                const theme = this.themes[choice.theme];
                // Apply theme colors to CSS variables for clean styling
                btn.style.setProperty('--theme-primary', theme.visual_style.primary_color);
                btn.style.setProperty('--theme-accent', theme.visual_style.accent_color);
                btn.setAttribute('role', 'button');
                btn.setAttribute('tabindex', '0');
                btn.setAttribute('aria-label', `Branch to subtheme: ${choice.subName} in ${choice.theme}`);
                btn.addEventListener('click', () => this.handleChoice(choice.theme, choice.subName));
                btn.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                        this.handleChoice(choice.theme, choice.subName);
                    }
                });
                choicesDiv.appendChild(btn);
            }, index * 800); // 800ms delay between each button appearance
        });
    }

    async handleChoice(themeName, subName) {
        // Scroll to top of page when choice is made
        window.scrollTo({ top: 0, behavior: 'smooth' });
        
        // Stop current processes
        this.stopCurrentProcesses();
        
        const viewport = document.getElementById('story-viewport');
        
        // Clear existing content more carefully
        const existingChoices = viewport.querySelector('.branching-choices');
        if (existingChoices) existingChoices.remove();
        
        // Clear text content but preserve the container
        let textContent = viewport.querySelector('#text-content');
        if (!textContent) {
            textContent = document.createElement('div');
            textContent.id = 'text-content';
            viewport.appendChild(textContent);
        } else {
            textContent.innerHTML = '';
        }
        
        this.currentNarrative = await this.compileNarrative(themeName, subName);
        this.currentNarrative.currentSegmentIndex = 0;
        this.applyThemeStyle(this.currentNarrative.visual_style);
        atmosphere.setMood(this.currentNarrative.visual_style.mood);
        this.typewriter = new ArtisticTypewriter(textContent, { speed: 30, questionSpeed: 15, reactionSpeed: 45, pauseAfterSentence: 500, pauseAfterParagraph: 1000 });
        this.typewriter.play();
        this.isPlaying = true;
        await this.playNextSnippet();
    }
    
    applyThemeStyle(style) {
        document.documentElement.style.setProperty('--theme-primary', style.primary_color);
        document.documentElement.style.setProperty('--theme-accent', style.accent_color);
        // Preserve dark mode state when applying theme
        const isDarkMode = document.body.classList.contains('dark');
        document.body.className = `theme-${style.mood}`;
        if (isDarkMode) {
            document.body.classList.add('dark');
        }
    }
    
    setupEventListeners() {
        const playPause = document.getElementById('play-pause');
        if (playPause) playPause.addEventListener('click', () => this.togglePlayPause());
        const nextStory = document.getElementById('next-story');
        if (nextStory) nextStory.addEventListener('click', () => this.nextNarrative());
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) themeToggle.addEventListener('click', () => {
            document.body.classList.toggle('dark');
            localStorage.setItem('darkMode', document.body.classList.contains('dark'));
        });
        
        // Sidebar toggle functionality
        const themesSidebarBtn = document.getElementById('themes-sidebar-btn');
        const subthemesSidebarBtn = document.getElementById('subthemes-sidebar-btn');
        const themesToggle = document.getElementById('themes-toggle');
        const subthemesToggle = document.getElementById('subthemes-toggle');
        
        if (themesSidebarBtn) {
            themesSidebarBtn.addEventListener('click', () => this.toggleSidebar('themes-sidebar'));
        }
        if (subthemesSidebarBtn) {
            subthemesSidebarBtn.addEventListener('click', () => this.toggleSidebar('subthemes-sidebar'));
        }
        if (themesToggle) {
            themesToggle.addEventListener('click', () => this.toggleSidebar('themes-sidebar'));
        }
        if (subthemesToggle) {
            subthemesToggle.addEventListener('click', () => this.toggleSidebar('subthemes-sidebar'));
        }
        
        // Keyboard nav for themes
        const themes = document.querySelectorAll('.theme-tag');
        themes.forEach((theme, idx) => {
            theme.addEventListener('focus', () => {
                const sr = document.getElementById('sr-announcements');
                sr.textContent = `Theme ${idx + 1} of ${themes.length}: ${theme.textContent}`;
            });
        });
        const backBtn = document.getElementById('back-to-themes');
        if (backBtn) backBtn.addEventListener('click', () => {
            document.getElementById('theme-constellation').style.display = 'flex';
            document.getElementById('story-viewport').classList.remove('active');
            document.getElementById('story-viewport').innerHTML = '';
        });
        const shareBtn = document.getElementById('share-insight');
        if (shareBtn) shareBtn.addEventListener('click', () => {
            const current = this.currentNarrative;
            const text = `${current.segments[0].text} ${current.segments[1].text}`;
            navigator.clipboard.writeText(text).then(() => alert('Insight copied to clipboard!'));
        });
    }
    
    toggleSidebar(sidebarId) {
        const sidebar = document.getElementById(sidebarId);
        if (sidebar) {
            sidebar.classList.toggle('open');
            
            // Update button text based on state
            const isOpen = sidebar.classList.contains('open');
            const toggleBtn = document.getElementById(sidebarId.replace('-sidebar', '-toggle'));
            const controlBtn = document.getElementById(sidebarId.replace('-sidebar', '-sidebar-btn'));
            
            if (toggleBtn) {
                toggleBtn.textContent = isOpen ? '×' : '☰';
            }
            
            // Update control button text
            if (controlBtn) {
                if (sidebarId === 'themes-sidebar') {
                    controlBtn.textContent = isOpen ? '📚' : '📖';
                } else if (sidebarId === 'subthemes-sidebar') {
                    controlBtn.textContent = isOpen ? '🔍' : '🔎';
                }
            }
            
            // Add visual feedback
            if (isOpen) {
                sidebar.style.boxShadow = '0 0 30px rgba(0, 0, 0, 0.2)';
            } else {
                sidebar.style.boxShadow = '0 0 20px rgba(0, 0, 0, 0.1)';
            }
        }
    }
    
    // Add loading states for better UX
    showLoadingState(element) {
        if (element) {
            element.style.opacity = '0.6';
            element.style.pointerEvents = 'none';
        }
    }
    
    hideLoadingState(element) {
        if (element) {
            element.style.opacity = '1';
            element.style.pointerEvents = 'auto';
        }
    }
    
    togglePlayPause() {
        if (this.typewriter.isPlaying) {
            this.typewriter.pause();
            document.getElementById('play-pause').textContent = '▶️';
            document.getElementById('play-pause').setAttribute('aria-label', 'Play narrative');
        } else {
            this.typewriter.play();
            document.getElementById('play-pause').textContent = '⏸️';
            document.getElementById('play-pause').setAttribute('aria-label', 'Pause narrative');
        }
    }
    
    nextNarrative() {
        // Logic to select and play next random theme
        const themeNames = Object.keys(this.themes);
        const randomTheme = themeNames[Math.floor(Math.random() * themeNames.length)];
        this.playNarrative(randomTheme);
    }
    
    async compileNarrative(themeName, subName) {
        let theme = this.themes[themeName];
        theme.segments = [];
        const sub = theme.subthemes.find(s => s.name === subName) || theme.subthemes[0];
        theme.segments.push({text: `<div class='question'>QUESTION <br>${sub.question}</div>`, pause_after: 500});  
        // each response has its own reaction
        const randomResponseReaction = sub.response_reactions[Math.floor(Math.random() * sub.response_reactions.length)];
        theme.segments.push({text: `<div class='answer'>RESPONSE <br>${randomResponseReaction.response}</div>`, pause_after: 1000});
        theme.segments.push({text: `<div class='reaction'>REACTION <br>${randomResponseReaction.reaction}</div>`, pause_after: 1500});
        
        return theme;
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Create the narrative engine instance but don't auto-initialize
    window.narrativeEngine = new NarrativeEngine();
    
    // Load dark mode preference
    if (localStorage.getItem('darkMode') === 'true') {
        document.body.classList.add('dark');
    }
}); 
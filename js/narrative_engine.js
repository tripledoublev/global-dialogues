class NarrativeEngine {
    constructor() {
        this.themes = {};
        this.currentNarrative = null;
        this.typewriter = null;
        this.init();
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
            this.themes = await response.json();
            this.renderThemeSelector();
        } catch (error) {
            console.error('Error loading themes:', error);
        }
    }
    
    renderThemeSelector() {
        const container = document.getElementById('theme-constellation');
        if (!container) return;
        
        container.innerHTML = '';
        const sortedThemes = Object.entries(this.themes).sort((a,b) => b[1].subthemes.length - a[1].subthemes.length).slice(0,5);
        sortedThemes.forEach(([themeName, theme]) => {
            const themeEl = document.createElement('div');
            themeEl.className = 'theme-tag';
            themeEl.textContent = themeName;
            themeEl.style.background = `linear-gradient(135deg, ${theme.visual_style.primary_color}, ${theme.visual_style.accent_color})`;
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
    }
    
    async playNarrative(themeName) {
        document.getElementById('theme-constellation').style.display = 'none';
        this.currentNarrative = await this.compileNarrative(themeName);
        const viewport = document.getElementById('story-viewport');
        const existingChoices = viewport.querySelector('.branching-choices');
        if (existingChoices) existingChoices.remove();
        viewport.classList.add('active');
        let textContent = viewport.querySelector('#text-content');
        if (!textContent) {
            textContent = document.createElement('div');
            textContent.id = 'text-content';
            viewport.appendChild(textContent);
        }
        textContent.innerHTML = '';
        this.applyThemeStyle(this.currentNarrative.visual_style);
        atmosphere.setMood(this.currentNarrative.visual_style.mood);
        this.typewriter = new ArtisticTypewriter(textContent, { speed: 30, questionSpeed: 15, pauseAfterSentence: 500, pauseAfterParagraph: 1000 });
        this.typewriter.play();
        await this.playNextSnippet();
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
        const otherThemes = Object.keys(this.themes).filter(t => t !== this.currentNarrative.themeName).sort(() => Math.random() - 0.5).slice(0,3);
        const subChoices = [];
        otherThemes.forEach(themeName => {
            const sub = this.themes[themeName].subthemes[Math.floor(Math.random() * this.themes[themeName].subthemes.length)];
            subChoices.push({theme: themeName, subName: sub.name});
        });
        subChoices.forEach(choice => {
            const btn = document.createElement('div');
            btn.className = 'theme-tag choice-btn';
            btn.textContent = choice.subName;
            const theme = this.themes[choice.theme];
            const angle = Math.floor(Math.random() * 90) + 90;
            const blendFactor = Math.random();
            const midColor = '#' + Math.floor((parseInt(theme.visual_style.primary_color.slice(1),16) * blendFactor + parseInt(theme.visual_style.accent_color.slice(1),16) * (1 - blendFactor))).toString(16);
            const thirdColor = '#' + Math.floor(Math.random()*16777215).toString(16); // Random color
            btn.style.background = `linear-gradient(${angle}deg, ${theme.visual_style.primary_color}, ${midColor}, ${thirdColor}, ${theme.visual_style.accent_color})`;
            const animSpeed = (Math.random() * 4 + 3).toFixed(1) + 's';
            btn.style.setProperty('--anim-speed', animSpeed);
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
        });
        viewport.appendChild(choicesDiv);
    }

    async handleChoice(themeName, subName) {
        const viewport = document.getElementById('story-viewport');
        viewport.querySelector('.branching-choices').remove();
        viewport.innerHTML = '';
        let textContent = document.createElement('div');
        textContent.id = 'text-content';
        viewport.appendChild(textContent);
        this.currentNarrative = await this.compileNarrative(themeName, subName);
        this.currentNarrative.currentSegmentIndex = 0;
        this.applyThemeStyle(this.currentNarrative.visual_style);
        atmosphere.setMood(this.currentNarrative.visual_style.mood);
        this.typewriter = new ArtisticTypewriter(textContent, { speed: 30, questionSpeed: 15, pauseAfterSentence: 500, pauseAfterParagraph: 1000 });
        this.typewriter.play();
        await this.playNextSnippet();
    }
    
    applyThemeStyle(style) {
        document.documentElement.style.setProperty('--theme-primary', style.primary_color);
        document.documentElement.style.setProperty('--theme-accent', style.accent_color);
        document.body.className = `theme-${style.mood}`;
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
        theme.segments.push({text: `<div class='question'>${sub.question}</div>`, pause_after: 500});
        const randomSnippet = sub.snippets[Math.floor(Math.random() * sub.snippets.length)];
        theme.segments.push({text: `<div class='answer'>${randomSnippet}</div>`, pause_after: 1000});
        return theme;
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    new NarrativeEngine();
    
    // Load dark mode preference
    if (localStorage.getItem('darkMode') === 'true') {
        document.body.classList.add('dark');
    }
}); 
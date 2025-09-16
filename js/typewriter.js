class ArtisticTypewriter {
  constructor(element, options = {}) {
    this.element = element;
    this.speed = options.speed || 25; // ms per character for answers
    this.questionSpeed = options.questionSpeed || (this.speed / 3); // 3x faster for questions
    this.reactionSpeed = options.reactionSpeed || (this.speed / 2); // 2x faster for reactions
    this.pauseAfterSentence = options.pauseAfterSentence || 500;
    this.pauseAfterParagraph = options.pauseAfterParagraph || 1000;
    this.isPlaying = false;
    this.currentText = '';
    this.queue = [];
    this.progressCallback = options.progressCallback || (() => {});
    this.announceCallback = options.announceCallback || (() => {});
    // cancellation and timers tracking
    this._cancelled = false;
    this._currentTimeout = null;
  }

  async typeText(html) {
    if (this._cancelled) return; // bail fast if cancelled
    // Create a temporary container to parse the HTML
    const temp = document.createElement('div');
    temp.innerHTML = html;
    
    // If the HTML contains a div with class, use it directly
    const hasDivWithClass = html.includes('<div class=');
    if (hasDivWithClass) {
      // Clone the parsed element and append it
      const clonedElement = temp.firstElementChild.cloneNode(true);
      this.element.appendChild(clonedElement);
      
      // Get the text content for typewriter effect
      const content = clonedElement.textContent;
      let typed = '';
      
      // Determine if this is a question, answer, or reaction based on the class
      const isQuestion = clonedElement.classList.contains('question');
      const isReaction = clonedElement.classList.contains('reaction');
      let currentSpeed = this.speed; // default speed for answers
      
      if (isQuestion) {
        currentSpeed = this.questionSpeed;
      } else if (isReaction) {
        currentSpeed = this.reactionSpeed;
      }
      
      // Clear the element and type the content
      clonedElement.textContent = '';
      for (let char of content) {
        if (this._cancelled || !this.isPlaying) break;
        typed += char;
        clonedElement.textContent = typed;
        await this.sleep(currentSpeed);
      }
    } else {
      // Fallback to the original span-based approach
      const isQuestion = html.includes('class=\'question\'');
      const isReaction = html.includes('class=\'reaction\'');
      let spanClass = 'answer'; // default
      let currentSpeed = this.speed; // default speed
      
      if (isQuestion) {
        spanClass = 'question';
        currentSpeed = this.questionSpeed;
      } else if (isReaction) {
        spanClass = 'reaction';
        currentSpeed = this.reactionSpeed;
      }
      
      const content = temp.textContent;
      const styledSpan = document.createElement('span');
      styledSpan.className = spanClass;
      this.element.appendChild(styledSpan);
      let typed = '';
      
      for (let char of content) {
        if (this._cancelled || !this.isPlaying) break;
        typed += char;
        styledSpan.textContent = typed;
        await this.sleep(currentSpeed);
      }
    }
  }

  formatText(text) {
    return text
      .replace(/"([^"]+)"/g, '&lt;em class=&quot;quote&quot;&gt;"$1"&lt;/em&gt;')
      .replace(/\*([^*]+)\*/g, '&lt;strong class=&quot;emphasis&quot;&gt;$1&lt;/strong&gt;')
      .replace(/\n/g, '&lt;br&gt;&lt;br&gt;');
  }

  isEmotional(word) {
    const emotionalWords = ['hope', 'fear', 'love', 'worry', 'dream', 'concern'];
    return emotionalWords.includes(word.toLowerCase());
  }

  play() { this._cancelled = false; this.isPlaying = true; }
  pause() { this.isPlaying = false; }
  cancel() {
    // fully cancel any ongoing animations/sleeps
    this._cancelled = true;
    this.isPlaying = false;
    if (this._currentTimeout) {
      clearTimeout(this._currentTimeout.id);
      // resolve the pending promise early if available
      if (this._currentTimeout.resolve) {
        this._currentTimeout.resolve();
      }
      this._currentTimeout = null;
    }
  }
  setSpeed(speed) { this.speed = speed; }
  
  sleep(ms) {
    if (this._cancelled || !this.isPlaying) return Promise.resolve();
    return new Promise(resolve => {
      const id = setTimeout(() => {
        // if cancelled during the timeout, resolve anyway to allow callers to bail
        resolve();
        // clear reference after completion
        if (this._currentTimeout && this._currentTimeout.id === id) {
          this._currentTimeout = null;
        }
      }, ms);
      this._currentTimeout = { id, resolve };
    });
  }
}

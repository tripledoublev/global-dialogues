class ArtisticTypewriter {
  constructor(element, options = {}) {
    this.element = element;
    this.speed = options.speed || 30; // ms per character for answers
    this.questionSpeed = options.questionSpeed || (this.speed / 2); // 2x faster for questions
    this.pauseAfterSentence = options.pauseAfterSentence || 500;
    this.pauseAfterParagraph = options.pauseAfterParagraph || 1000;
    this.isPlaying = false;
    this.currentText = '';
    this.queue = [];
    this.progressCallback = options.progressCallback || (() => {});
    this.announceCallback = options.announceCallback || (() => {});
  }

  async typeText(html) {
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
      
      // Determine if this is a question or answer based on the class
      const isQuestion = clonedElement.classList.contains('question');
      const currentSpeed = isQuestion ? this.questionSpeed : this.speed;
      
      // Clear the element and type the content
      clonedElement.textContent = '';
      for (let char of content) {
        typed += char;
        clonedElement.textContent = typed;
        await this.sleep(currentSpeed);
      }
    } else {
      // Fallback to the original span-based approach
      const isQuestion = html.includes('class=\'question\'');
      const spanClass = isQuestion ? 'question' : 'answer';
      const content = temp.textContent;
      const styledSpan = document.createElement('span');
      styledSpan.className = spanClass;
      this.element.appendChild(styledSpan);
      let typed = '';
      
      // Use appropriate speed based on whether it's a question or answer
      const currentSpeed = isQuestion ? this.questionSpeed : this.speed;
      
      for (let char of content) {
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

  play() { this.isPlaying = true; }
  pause() { this.isPlaying = false; }
  setSpeed(speed) { this.speed = speed; }
  
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
} 
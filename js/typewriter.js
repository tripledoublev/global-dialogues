class ArtisticTypewriter {
  constructor(element, options = {}) {
    this.element = element;
    this.speed = options.speed || 30; // ms per character
    this.pauseAfterSentence = options.pauseAfterSentence || 500;
    this.pauseAfterParagraph = options.pauseAfterParagraph || 1000;
    this.isPlaying = false;
    this.currentText = '';
    this.queue = [];
    this.progressCallback = options.progressCallback || (() => {});
    this.announceCallback = options.announceCallback || (() => {});
  }

  async typeText(html) {
    const temp = document.createElement('span');
    temp.innerHTML = html;
    const isQuestion = html.includes('class=\'question\'');
    const spanClass = isQuestion ? 'question' : 'answer';
    const content = temp.textContent;
    const styledSpan = document.createElement('span');
    styledSpan.className = spanClass;
    this.container.appendChild(styledSpan);
    let typed = '';
    for (let char of content) {
        typed += char;
        styledSpan.textContent = typed;
        await this.sleep(this.options.speed);
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
document.addEventListener("DOMContentLoaded", async () => {
  const outputDiv = document.getElementById("output");
  const API_KEY = "AIzaSyDZ5O1Wv-0E5deA9WPJAFEZlfEkVIOUmfs";
  const API_URL = "http://3.110.185.110:8000"; // Fix: removed double semicolon and /predict suffix

  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const url = tabs[0].url;
    const youtubeRegex = /^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/;
    const match = url.match(youtubeRegex);

    if (match && match[1]) {
      const videoId = match[1];
      outputDiv.innerHTML = `<div class="section-title">YouTube Video ID</div><p>${videoId}</p><p>Fetching comments...</p>`;

      const comments = await fetchComments(videoId);
      if (comments.length === 0) {
        outputDiv.innerHTML += "<p>No comments found for this video.</p>";
        return;
      }

      outputDiv.innerHTML += `<p>Fetched ${comments.length} comments. Performing sentiment analysis...</p>`;
      const result = await getSentimentPredictions(comments);

      if (result && result.success) {
        const predictions = result.predictions;
        const metrics = result.metrics;

        // Fix: use metrics from API response directly
        const sentimentCounts = {
          "1": metrics.positive,
          "0": metrics.neutral,
          "-1": metrics.negative
        };

        const totalSentimentScore = predictions.reduce(
          (sum, item) => sum + parseInt(item.sentiment), 0
        );
        const avgSentimentScore = (totalSentimentScore / predictions.length).toFixed(2);
        const normalizedSentimentScore = (
          ((parseFloat(avgSentimentScore) + 1) / 2) * 10
        ).toFixed(2);

        const uniqueCommenters = new Set(
          comments.map(comment => comment.authorId)
        ).size;
        const totalWords = comments.reduce(
          (sum, comment) =>
            sum + comment.text.split(/\s+/).filter(w => w.length > 0).length,
          0
        );
        const avgWordLength = (totalWords / metrics.total_comments).toFixed(2);

        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-title">Comment Analysis Summary</div>
            <div class="metrics-container">
              <div class="metric">
                <div class="metric-title">Total Comments</div>
                <div class="metric-value">${metrics.total_comments}</div>
              </div>
              <div class="metric">
                <div class="metric-title">Unique Commenters</div>
                <div class="metric-value">${uniqueCommenters}</div>
              </div>
              <div class="metric">
                <div class="metric-title">Avg Comment Length</div>
                <div class="metric-value">${avgWordLength} words</div>
              </div>
              <div class="metric">
                <div class="metric-title">Avg Sentiment Score</div>
                <div class="metric-value">${normalizedSentimentScore}/10</div>
              </div>
            </div>
          </div>`;

        // Sentiment chart from base64
        const chartBase64 = result.visualizations.sentiment_donut_chart;
        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-title">Sentiment Analysis Results</div>
            <p>Positive: ${metrics.positive} | Neutral: ${metrics.neutral} | Negative: ${metrics.negative}</p>
            ${chartBase64
              ? `<img src="data:image/png;base64,${chartBase64}" style="width:100%;margin-top:10px;" />`
              : '<p>Chart not available.</p>'}
          </div>`;

        // Wordcloud from base64
        const wordcloudBase64 = result.visualizations.wordcloud_chart;
        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-title">Comment Wordcloud</div>
            ${wordcloudBase64
              ? `<img src="data:image/png;base64,${wordcloudBase64}" style="width:100%;margin-top:10px;" />`
              : '<p>Wordcloud not available.</p>'}
          </div>`;

        // Trend graph
        const sentimentData = predictions.map(item => ({
          timestamp: item.timestamp,
          sentiment: parseInt(item.sentiment)
        }));
        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-title">Sentiment Trend Over Time</div>
            <div id="trend-graph-container"><p>Loading trend graph...</p></div>
          </div>`;
        await fetchAndDisplayTrendGraph(sentimentData);

        // Top comments
        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-title">Top 25 Comments with Sentiments</div>
            <ul class="comment-list">
              ${predictions.slice(0, 25).map((item, index) => `
                <li class="comment-item">
                  <span>${index + 1}. ${item.comment}</span><br>
                  <span class="comment-sentiment">Sentiment: ${item.sentiment}</span>
                </li>`).join('')}
            </ul>
          </div>`;
      }
    } else {
      outputDiv.innerHTML = "<p>This is not a valid YouTube URL.</p>";
    }
  });

  async function fetchComments(videoId) {
    let comments = [];
    let pageToken = "";
    try {
      while (comments.length < 500) {
        const response = await fetch(
          `https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=100&pageToken=${pageToken}&key=${API_KEY}`
        );
        const data = await response.json();
        if (data.items) {
          data.items.forEach(item => {
            const commentText = item.snippet.topLevelComment.snippet.textOriginal;
            const timestamp = item.snippet.topLevelComment.snippet.publishedAt;
            const authorId = item.snippet.topLevelComment.snippet.authorChannelId?.value || 'Unknown';
            comments.push({ text: commentText, timestamp, authorId });
          });
        }
        pageToken = data.nextPageToken;
        if (!pageToken) break;
      }
    } catch (error) {
      console.error("Error fetching comments:", error);
      outputDiv.innerHTML += "<p>Error fetching comments.</p>";
    }
    return comments;
  }

  async function getSentimentPredictions(comments) {
    try {
      const cleanedComments = comments.map(c => ({
        text: c.text,
        published_at: c.timestamp  // Fix: match API schema field name
      }));

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments: cleanedComments })
      });

      const result = await response.json();
      if (response.ok) {
        return result;
      } else {
        throw new Error(result.detail || 'Error fetching predictions');
      }
    } catch (error) {
      console.error("Error fetching predictions:", error);
      outputDiv.innerHTML += "<p>Error fetching sentiment predictions.</p>";
      return null;
    }
  }

  async function fetchAndDisplayTrendGraph(sentimentData) {
    const trendGraphContainer = document.getElementById('trend-graph-container');
    try {
      const response = await fetch(`${API_URL}/generate_trend_graph`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentiment_data: sentimentData })
      });
      if (!response.ok) throw new Error('Failed to fetch trend graph');
      const blob = await response.blob();
      const imgURL = URL.createObjectURL(blob);
      trendGraphContainer.innerHTML = '';
      const img = document.createElement('img');
      img.src = imgURL;
      img.style.width = '100%';
      img.style.marginTop = '20px';
      trendGraphContainer.appendChild(img);
    } catch (error) {
      console.error("Error fetching trend graph:", error);
      trendGraphContainer.innerHTML = "<p style='color:red;'>Error fetching trend graph.</p>";
    }
  }
});
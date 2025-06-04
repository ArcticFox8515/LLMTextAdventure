import React, { useState, useEffect } from 'react';
import './App.css';
import FeedbackPopup from './FeedbackPopup';

interface AdventureUserInput {
  action?: string;
  instructions?: string;
  error?: string;
  errorDetails?: string[];
}

interface AdventureTurnViewModel {
  turnNumber: number;
  narrative: string;
  suggestedActions: string;
  userInput?: AdventureUserInput;
}

interface ImageData {
  content: string[];
  role: 'player' | 'background' | 'illustration';
}

const App: React.FC = () => {
  const [query, setQuery] = useState('');
  const [turnInfo, setTurnInfo] = useState<AdventureTurnViewModel[]>([]);
  const [currentMessageIndex, setCurrentMessageIndex] = useState(0); // Track current message index
  const [turnInProgress, setTurnInProgress] = useState(false);
  const [ws, setWs] = useState<WebSocket | null>(null);

  // Image states
  const [playerImage, setPlayerImage] = useState<string>('');
  const [backgroundImage, setBackgroundImage] = useState<string>('');
  const [illustrationImage, setIllustrationImage] = useState<string>('');

  // Feedback popup state
  const [feedbackPopupOpen, setFeedbackPopupOpen] = useState(false);
  const [feedbackType, setFeedbackType] = useState<'like' | 'dislike' | null>(null);
  useEffect(() => {
    // For Electron, connect directly to localhost WebSocket
    const wsUrl = 'ws://localhost:3002';
    const socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      console.log('WebSocket connection established');
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'turn-update') {
        const turnInfo: AdventureTurnViewModel = data.content;
        setTurnInfo((prev) => {
          let updated = [...prev];
          while (updated.length <= turnInfo.turnNumber) {
            updated.push({ turnNumber: updated.length, narrative: '', suggestedActions: '' });
            setCurrentMessageIndex(updated.length - 1); // Automatically switch to the latest message
          }
          updated[turnInfo.turnNumber] = turnInfo;
          return updated;
        });
      }
      else if (data.type === 'llm-running') {
        setTurnInProgress(data.content);
      }
      else if (data.type === 'image-update') {
        const imageData: ImageData = data;
        if (imageData.content && imageData.content.length > 0) {
          const imageUrl = `data:image/png;base64,${imageData.content[0]}`;

          switch (imageData.role) {
            case 'player':
              setPlayerImage(imageUrl);
              break;
            case 'background':
              setBackgroundImage(imageUrl);
              break;
            case 'illustration':
              setIllustrationImage(imageUrl);
              break;
          }
        }
      }
    };

    socket.onclose = () => {
      console.log('WebSocket connection closed');
    };

    setWs(socket);

    return () => {
      socket.close();
    };
  }, []);

  const handleSendMessage = () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'action', characterAction: query }));
      setQuery('');
    } else {
      console.error('WebSocket is not open');
    }
  };

  const handleSendFeedback = (type: 'like' | 'dislike', comment: string) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'feedback',
        feedbackType: type,
        feedbackComment: comment
      }));
    } else {
      console.error('WebSocket is not open');
    }
  };

  const handleFeedbackButtonClick = (type: 'like' | 'dislike') => {
    setFeedbackType(type);
    setFeedbackPopupOpen(true);
  };

  const handleRefreshImage = (imageRole: 'player' | 'background' | 'illustration') => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'refresh-image', role: imageRole }));
    } else {
      console.error('WebSocket is not open');
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.shiftKey && event.key === 'Enter') {
      event.preventDefault(); // Prevent newline insertion
      handleSendMessage();
    }
  };

  const handlePreviousMessage = () => {
    setCurrentMessageIndex((prevIndex) => Math.max(prevIndex - 1, 0));
  };

  const handleNextMessage = () => {
    setCurrentMessageIndex((prevIndex) => Math.min(prevIndex + 1, turnInfo.length - 1));
  };

  const getCurrentTurnInfo = (): AdventureTurnViewModel => {
    return currentMessageIndex < turnInfo.length ? turnInfo[currentMessageIndex] : { turnNumber: 0, narrative: '', suggestedActions: '' };
  }

  return (
    <div className="app-container">
      {/* Background Image */}
      {backgroundImage && (
        <div className="background-image" style={{ backgroundImage: `url(${backgroundImage})` }}></div>
      )}

      <div className="content-container">
        {/* Player Image */}
        <div className="player-image-container">
          {playerImage && (
            <div className="player-image-wrapper">
              <div className="player-image" style={{ backgroundImage: `url(${playerImage})` }}></div>
              <button
                className="image-refresh-button"
                onClick={() => handleRefreshImage('player')}
                title="Refresh player image"
              >
                ↻
              </button>
            </div>
          )}
        </div>

        <div className="center-content">
          <div className="message-container">
            <div>
              {getCurrentTurnInfo().userInput?.action && (<p>&gt;&nbsp;{getCurrentTurnInfo().userInput?.action}</p>)}
              <p>{getCurrentTurnInfo().narrative}</p>
              {(getCurrentTurnInfo().suggestedActions).split('\n').map((action, index) => (
                <p key={index}>
                  <a key={index} onClick={() => setQuery(action)} style={{ cursor: 'pointer' }}>
                    {action}
                  </a>
                </p>
              ))}
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ flex: '1' }}></div>
            <div className="navigation-buttons">
              <button onClick={handlePreviousMessage} disabled={currentMessageIndex === 0}>
                Previous
              </button>
              <span>{currentMessageIndex}</span>
              <button onClick={handleNextMessage} disabled={currentMessageIndex >= turnInfo.length - 1}>
                Next
              </button>
            </div>
            <div style={{ flex: '1', display: 'flex', justifyContent: 'flex-end' }}>
              {/* Feedback buttons */}
              <div className="feedback-buttons">
                <button
                  className="feedback-button feedback-button-like"
                  onClick={() => handleFeedbackButtonClick('like')}
                >
                  👍
                </button>
                <button
                  className="feedback-button feedback-button-dislike"
                  onClick={() => handleFeedbackButtonClick('dislike')}
                >
                  👎
                </button>
              </div>
            </div>
          </div>
          <textarea
            className="textarea"
            placeholder="Type your message here..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <button className="send-button" onClick={handleSendMessage} disabled={turnInProgress}>
            Send Message
          </button>
        </div>

        {/* Illustration Image */}
        <div className="illustration-image-container">
          {illustrationImage && (
            <div className="illustration-image-wrapper">
              <div className="illustration-image" style={{ backgroundImage: `url(${illustrationImage})` }}></div>
              <button
                className="image-refresh-button"
                onClick={() => handleRefreshImage('illustration')}
                title="Refresh illustration"
              >
                ↻
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Feedback popup */}
      <FeedbackPopup
        isOpen={feedbackPopupOpen}
        feedbackType={feedbackType}
        onClose={() => setFeedbackPopupOpen(false)}
        onSubmit={handleSendFeedback}
      />
    </div>
  );
};

export default App;

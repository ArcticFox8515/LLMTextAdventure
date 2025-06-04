import React, { useState } from 'react';

interface FeedbackPopupProps {
  isOpen: boolean;
  feedbackType: 'like' | 'dislike' | null;
  onClose: () => void;
  onSubmit: (type: 'like' | 'dislike', comment: string) => void;
}

const FeedbackPopup: React.FC<FeedbackPopupProps> = ({ 
  isOpen, 
  feedbackType, 
  onClose,
  onSubmit
}) => {
  const [comment, setComment] = useState('');

  if (!isOpen || !feedbackType) return null;

  const handleSubmit = () => {
    onSubmit(feedbackType, comment);
    setComment('');
    onClose();
  };

  return (
    <div className="feedback-popup-overlay">
      <div className="feedback-popup">
        <div className="feedback-popup-header">
          <h3>{feedbackType === 'like' ? 'What did you like?' : 'What could be improved?'}</h3>
          <button className="feedback-close-button" onClick={onClose}>×</button>
        </div>
        <textarea
          value={comment}
          onChange={(e) => setComment(e.target.value)}
          placeholder="Enter your feedback here..."
          className="feedback-textarea"
        />
        <button className="feedback-submit-button" onClick={handleSubmit}>
          Send Feedback
        </button>
      </div>
    </div>
  );
};

export default FeedbackPopup;
/**
 * CommentThread Component
 *
 * Comment thread UI with:
 * - Nested replies
 * - Rich text editor (Markdown)
 * - @mention autocomplete
 * - Emoji reactions
 * - Comment resolution
 *
 * @module CommentThread
 */

import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import {
  Comment,
  CommentThread as CommentThreadType,
  User,
  CommentReaction,
} from '../../../services/collaboration/types';

// Emoji picker (simplified)
const EMOJI_LIST = ['👍', '❤️', '😄', '🎉', '👀', '🚀', '👏', '💯'];

interface CommentThreadProps {
  thread: CommentThreadType;
  currentUser: User;
  onReply: (content: string) => void;
  onResolve: () => void;
  onReact?: (commentId: string, emoji: string) => void;
  onDelete?: (commentId: string) => void;
}

/**
 * Main CommentThread component
 */
export const CommentThread: React.FC<CommentThreadProps> = ({
  thread,
  currentUser,
  onReply,
  onResolve,
  onReact,
  onDelete,
}) => {
  const [showReplyBox, setShowReplyBox] = useState(false);
  const [replyContent, setReplyContent] = useState('');
  const [showEmojiPicker, setShowEmojiPicker] = useState<string | null>(null);
  const [editingCommentId, setEditingCommentId] = useState<string | null>(null);

  const handleReply = useCallback(() => {
    if (!replyContent.trim()) return;

    onReply(replyContent);
    setReplyContent('');
    setShowReplyBox(false);
  }, [replyContent, onReply]);

  return (
    <div className={`comment-thread ${thread.resolved ? 'resolved' : ''}`}>
      {/* Root Comment */}
      <CommentItem
        comment={thread.rootComment}
        currentUser={currentUser}
        isRoot={true}
        onReact={onReact}
        onDelete={onDelete}
        showEmojiPicker={showEmojiPicker}
        setShowEmojiPicker={setShowEmojiPicker}
        editingCommentId={editingCommentId}
        setEditingCommentId={setEditingCommentId}
      />

      {/* Replies */}
      {thread.replies.length > 0 && (
        <div className="comment-replies">
          {thread.replies.map(reply => (
            <CommentItem
              key={reply.id}
              comment={reply}
              currentUser={currentUser}
              isRoot={false}
              onReact={onReact}
              onDelete={onDelete}
              showEmojiPicker={showEmojiPicker}
              setShowEmojiPicker={setShowEmojiPicker}
              editingCommentId={editingCommentId}
              setEditingCommentId={setEditingCommentId}
            />
          ))}
        </div>
      )}

      {/* Reply Box */}
      {!thread.resolved && (
        <div className="comment-actions">
          {!showReplyBox ? (
            <button
              onClick={() => setShowReplyBox(true)}
              className="reply-button"
            >
              Reply
            </button>
          ) : (
            <div className="reply-box">
              <MarkdownEditor
                value={replyContent}
                onChange={setReplyContent}
                placeholder="Write a reply... (Markdown supported)"
                currentUser={currentUser}
              />
              <div className="reply-actions">
                <button onClick={handleReply} className="submit-button">
                  Reply
                </button>
                <button
                  onClick={() => {
                    setShowReplyBox(false);
                    setReplyContent('');
                  }}
                  className="cancel-button"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          {!thread.resolved && (
            <button onClick={onResolve} className="resolve-button">
              ✓ Resolve
            </button>
          )}
        </div>
      )}

      {thread.resolved && (
        <div className="resolved-badge">
          ✓ Resolved by {thread.rootComment.resolvedBy?.name}
        </div>
      )}
    </div>
  );
};

/**
 * Individual Comment Item
 */
interface CommentItemProps {
  comment: Comment;
  currentUser: User;
  isRoot: boolean;
  onReact?: (commentId: string, emoji: string) => void;
  onDelete?: (commentId: string) => void;
  showEmojiPicker: string | null;
  setShowEmojiPicker: (id: string | null) => void;
  editingCommentId: string | null;
  setEditingCommentId: (id: string | null) => void;
}

const CommentItem: React.FC<CommentItemProps> = ({
  comment,
  currentUser,
  isRoot,
  onReact,
  onDelete,
  showEmojiPicker,
  setShowEmojiPicker,
  editingCommentId,
  setEditingCommentId,
}) => {
  const [editContent, setEditContent] = useState(comment.content);
  const emojiPickerRef = useRef<HTMLDivElement>(null);

  // Close emoji picker on outside click
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        emojiPickerRef.current &&
        !emojiPickerRef.current.contains(event.target as Node)
      ) {
        setShowEmojiPicker(null);
      }
    };

    if (showEmojiPicker === comment.id) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => {
        document.removeEventListener('mousedown', handleClickOutside);
      };
    }
  }, [showEmojiPicker, comment.id, setShowEmojiPicker]);

  const handleReaction = useCallback((emoji: string) => {
    if (onReact) {
      onReact(comment.id, emoji);
    }
    setShowEmojiPicker(null);
  }, [comment.id, onReact, setShowEmojiPicker]);

  const handleDelete = useCallback(() => {
    if (onDelete && window.confirm('Are you sure you want to delete this comment?')) {
      onDelete(comment.id);
    }
  }, [comment.id, onDelete]);

  const handleSaveEdit = useCallback(() => {
    // In production, this would call an API to update the comment
    console.log('Save edited comment:', editContent);
    setEditingCommentId(null);
  }, [editContent, setEditingCommentId]);

  // Group reactions by emoji
  const groupedReactions = useMemo(() => {
    const groups = new Map<string, CommentReaction[]>();

    comment.reactions.forEach(reaction => {
      if (!groups.has(reaction.emoji)) {
        groups.set(reaction.emoji, []);
      }
      groups.get(reaction.emoji)!.push(reaction);
    });

    return groups;
  }, [comment.reactions]);

  const formatTimestamp = (timestamp: string): string => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className={`comment-item ${isRoot ? 'root' : 'reply'}`}>
      <div className="comment-header">
        <div className="author-info">
          <div
            className="author-avatar"
            style={{ backgroundColor: comment.author.color }}
          >
            {comment.author.avatar ? (
              <img src={comment.author.avatar} alt={comment.author.name} />
            ) : (
              comment.author.name[0].toUpperCase()
            )}
          </div>
          <div>
            <div className="author-name">{comment.author.name}</div>
            <div className="comment-time">{formatTimestamp(comment.createdAt)}</div>
          </div>
        </div>

        <div className="comment-menu">
          {comment.author.id === currentUser.id && (
            <>
              <button
                onClick={() => setEditingCommentId(comment.id)}
                className="menu-button"
                title="Edit"
              >
                ✏️
              </button>
              <button
                onClick={handleDelete}
                className="menu-button"
                title="Delete"
              >
                🗑️
              </button>
            </>
          )}
        </div>
      </div>

      <div className="comment-content">
        {editingCommentId === comment.id ? (
          <div className="edit-box">
            <MarkdownEditor
              value={editContent}
              onChange={setEditContent}
              placeholder="Edit comment..."
              currentUser={currentUser}
            />
            <div className="edit-actions">
              <button onClick={handleSaveEdit} className="submit-button">
                Save
              </button>
              <button
                onClick={() => {
                  setEditingCommentId(null);
                  setEditContent(comment.content);
                }}
                className="cancel-button"
              >
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <ReactMarkdown className="markdown-content">
            {comment.content}
          </ReactMarkdown>
        )}
      </div>

      {/* Reactions */}
      <div className="comment-reactions">
        {Array.from(groupedReactions.entries()).map(([emoji, reactions]) => (
          <button
            key={emoji}
            className={`reaction-badge ${
              reactions.some(r => r.userId === currentUser.id) ? 'active' : ''
            }`}
            onClick={() => handleReaction(emoji)}
            title={reactions.map(r => r.userId).join(', ')}
          >
            {emoji} {reactions.length}
          </button>
        ))}

        {/* Add reaction button */}
        <div className="emoji-picker-container" ref={emojiPickerRef}>
          <button
            className="add-reaction-button"
            onClick={() =>
              setShowEmojiPicker(showEmojiPicker === comment.id ? null : comment.id)
            }
          >
            + 😊
          </button>

          {showEmojiPicker === comment.id && (
            <div className="emoji-picker">
              {EMOJI_LIST.map(emoji => (
                <button
                  key={emoji}
                  onClick={() => handleReaction(emoji)}
                  className="emoji-option"
                >
                  {emoji}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

/**
 * Markdown Editor Component with @mention support
 */
interface MarkdownEditorProps {
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
  currentUser: User;
  users?: User[];
}

const MarkdownEditor: React.FC<MarkdownEditorProps> = ({
  value,
  onChange,
  placeholder,
  currentUser,
  users = [],
}) => {
  const [showMentionSuggestions, setShowMentionSuggestions] = useState(false);
  const [mentionQuery, setMentionQuery] = useState('');
  const [cursorPosition, setCursorPosition] = useState(0);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = e.target.value;
    const cursorPos = e.target.selectionStart;

    onChange(newValue);
    setCursorPosition(cursorPos);

    // Check for @ mentions
    const textBeforeCursor = newValue.substring(0, cursorPos);
    const lastAtIndex = textBeforeCursor.lastIndexOf('@');

    if (lastAtIndex !== -1) {
      const textAfterAt = textBeforeCursor.substring(lastAtIndex + 1);
      if (!textAfterAt.includes(' ')) {
        setMentionQuery(textAfterAt);
        setShowMentionSuggestions(true);
      } else {
        setShowMentionSuggestions(false);
      }
    } else {
      setShowMentionSuggestions(false);
    }
  }, [onChange]);

  const handleMentionSelect = useCallback((user: User) => {
    if (!textareaRef.current) return;

    const textBeforeCursor = value.substring(0, cursorPosition);
    const lastAtIndex = textBeforeCursor.lastIndexOf('@');
    const textAfterCursor = value.substring(cursorPosition);

    const newValue =
      value.substring(0, lastAtIndex) +
      `@${user.name} ` +
      textAfterCursor;

    onChange(newValue);
    setShowMentionSuggestions(false);

    // Set cursor position after mention
    setTimeout(() => {
      if (textareaRef.current) {
        const newCursorPos = lastAtIndex + user.name.length + 2;
        textareaRef.current.setSelectionRange(newCursorPos, newCursorPos);
        textareaRef.current.focus();
      }
    }, 0);
  }, [value, cursorPosition, onChange]);

  const filteredUsers = useMemo(() => {
    if (!mentionQuery) return users;
    return users.filter(user =>
      user.name.toLowerCase().includes(mentionQuery.toLowerCase())
    );
  }, [users, mentionQuery]);

  return (
    <div className="markdown-editor">
      <textarea
        ref={textareaRef}
        value={value}
        onChange={handleChange}
        placeholder={placeholder}
        className="markdown-textarea"
      />

      {/* Mention suggestions */}
      {showMentionSuggestions && filteredUsers.length > 0 && (
        <div className="mention-suggestions">
          {filteredUsers.slice(0, 5).map(user => (
            <button
              key={user.id}
              onClick={() => handleMentionSelect(user)}
              className="mention-suggestion"
            >
              <div
                className="mention-avatar"
                style={{ backgroundColor: user.color }}
              >
                {user.avatar ? (
                  <img src={user.avatar} alt={user.name} />
                ) : (
                  user.name[0].toUpperCase()
                )}
              </div>
              <span>{user.name}</span>
            </button>
          ))}
        </div>
      )}

      {/* Preview */}
      {value && (
        <div className="markdown-preview">
          <h4>Preview:</h4>
          <ReactMarkdown>{value}</ReactMarkdown>
        </div>
      )}

      {/* Formatting help */}
      <div className="markdown-help">
        <small>
          <strong>Bold</strong> **text** | <em>Italic</em> *text* |{' '}
          <code>Code</code> `code` | @mention users
        </small>
      </div>
    </div>
  );
};

export default CommentThread;

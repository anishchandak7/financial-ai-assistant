import React from 'react';
import { marked } from 'marked';
import DOMPurify from 'dompurify';

export default function MarkdownRenderer({ content }) {
  if (!content) return null;
  const cleanHtml = DOMPurify.sanitize(marked.parse(content));
  return <div className="prose-google" dangerouslySetInnerHTML={{ __html: cleanHtml }} />;
}

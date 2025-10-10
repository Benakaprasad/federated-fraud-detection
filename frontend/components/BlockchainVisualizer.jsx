/*
BlockchainVisualizer.jsx - Displays blockchain chain in real-time
*/
import React from 'react';
export default function BlockchainVisualizer({ blocks }) {
  return (
    <div className="flex gap-2 overflow-x-auto">
      {blocks.map((b, i) => (
        <div key={i} className="p-2 border rounded-md bg-white">
          <p>Block #{b.index}</p>
          <p>Hash: {b.hash.slice(0,8)}...</p>
        </div>
      ))}
    </div>
  );
}

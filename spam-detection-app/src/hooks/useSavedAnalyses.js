import { useQuery } from '@tanstack/react-query';

/**
 * Custom hook for fetching saved analyses with search, sort, and pagination
 * @param {Object} options - Query options
 * @param {string} options.search - Search query to filter analyses
 * @param {string} options.sortBy - Field to sort by ('created_at' or 'id')
 * @param {string} options.sortOrder - Sort order ('asc' or 'desc')
 * @param {number} options.page - Current page number (1-indexed)
 * @param {number} options.pageSize - Number of items per page
 * @param {boolean} options.labMode - Whether lab mode is enabled
 * @param {boolean} options.enabled - Whether the query should run
 */
export function useSavedAnalyses({
  search = '',
  sortBy = 'created_at',
  sortOrder = 'desc',
  page = 1,
  pageSize = 10,
  labMode = false,
  enabled = true,
} = {}) {
  return useQuery({
    // Only include labMode in queryKey since that affects what data we fetch from the server
    // Search, sort, and pagination are client-side only
    queryKey: ['savedAnalyses', { labMode }],
    queryFn: async () => {
      const userId = getUserId();
      const resp = await fetch('http://localhost:8000/analysis/list', {
        headers: {
          'X-User-Id': userId,
          'X-Lab-Mode': labMode ? 'true' : 'false',
        },
      });

      if (!resp.ok) {
        throw new Error(`Failed to load saved analyses: ${resp.status}`);
      }

      const data = await resp.json();
      return data || [];
    },
    enabled,
    select: (data) => {
      // Client-side filtering, sorting, and pagination
      let filtered = [...data];

      // Apply search filter
      if (search.trim()) {
        const searchLower = search.toLowerCase();
        filtered = filtered.filter((item) => {
          const snippet = (item.snippet || '').toLowerCase();
          const createdAt = (item.created_at || '').toLowerCase();
          return snippet.includes(searchLower) || createdAt.includes(searchLower);
        });
      }

      // Apply sorting
      filtered.sort((a, b) => {
        let aValue, bValue;

        if (sortBy === 'created_at') {
          aValue = new Date(a.created_at || 0).getTime();
          bValue = new Date(b.created_at || 0).getTime();
        } else if (sortBy === 'id') {
          aValue = a.id || 0;
          bValue = b.id || 0;
        } else {
          // Default to created_at
          aValue = new Date(a.created_at || 0).getTime();
          bValue = new Date(b.created_at || 0).getTime();
        }

        if (sortOrder === 'asc') {
          return aValue - bValue;
        } else {
          return bValue - aValue;
        }
      });

      // Apply pagination
      const totalItems = filtered.length;
      const totalPages = Math.ceil(totalItems / pageSize);
      const startIndex = (page - 1) * pageSize;
      const endIndex = startIndex + pageSize;
      const paginatedData = filtered.slice(startIndex, endIndex);

      return {
        data: paginatedData,
        pagination: {
          page,
          pageSize,
          totalItems,
          totalPages,
          hasNextPage: page < totalPages,
          hasPreviousPage: page > 1,
        },
      };
    },
  });
}

// Helper function to get user ID from localStorage
// Matches the implementation in SpamDetector.jsx
function getUserId() {
  if (typeof window === 'undefined') return '';
  const username = localStorage.getItem('username');
  if (username) {
    return username;
  }
  
  // Fallback to generated user_id for unauthenticated users
  const STORAGE_KEY = 'hts_user_id';
  let userId = localStorage.getItem(STORAGE_KEY);
  if (!userId) {
    // Generate a simple user ID if it doesn't exist
    userId = `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    localStorage.setItem(STORAGE_KEY, userId);
  }
  return userId;
}


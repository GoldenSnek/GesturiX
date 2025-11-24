// File: frontend/utils/supabaseApi.ts
import { supabase } from '../src/supabaseClient';

/**
 * Returns the current authenticated user's id (if logged in)
 * @returns {Promise<string|null>} The user's uuid or null if not logged in
 */
export async function getCurrentUserId() {
  const { data, error } = await supabase.auth.getUser();
  if (error || !data?.user) return null;
  return data.user.id; // Supabase UUID, matches your DB schema
}

export async function fetchUserStatistics(userId: string) {
  const { data, error } = await supabase
    .from('user_statistics')
    .select('lessons_completed, days_streak, practice_hours')
    .eq('user_id', userId)
    .single();

  if (error || !data) {
    return {
      lessons_completed: 0,
      days_streak: 0,
      practice_hours: 0,
    };
  }
  return data;
}

// --- Leaderboard Functions ---

export async function fetchLeaderboard() {
  // Fetches ALL users ordered by lessons_completed.
  // We use !inner to exclude any stats records that might not have a matching profile.
  const { data, error } = await supabase
    .from('user_statistics')
    .select(`
      user_id,
      lessons_completed,
      days_streak,
      practice_hours,
      profiles!inner (
        username,
        photo_url
      )
    `)
    .order('lessons_completed', { ascending: false });

  if (error) {
    console.error('Error fetching leaderboard:', error);
    return [];
  }
  return data;
}

// --- Saved Items Functions ---

export interface SavedItem {
  id: string;
  item_type: 'letter' | 'number' | 'phrase';
  item_identifier: string;
  created_at: string;
}

export async function getUserSavedItems(userId: string): Promise<SavedItem[]> {
  const { data, error } = await supabase
    .from('user_saved_items')
    .select('*')
    .eq('user_id', userId);

  if (error) {
    console.error('Error fetching saved items:', error);
    return [];
  }
  return data || [];
}

export async function saveItem(userId: string, itemType: string, itemIdentifier: string) {
  // Check if already exists to avoid duplicates
  const { data: existing } = await supabase
    .from('user_saved_items')
    .select('id')
    .match({ user_id: userId, item_type: itemType, item_identifier: itemIdentifier })
    .maybeSingle();

  if (existing) return; // Already saved

  const { error } = await supabase
    .from('user_saved_items')
    .insert({ user_id: userId, item_type: itemType, item_identifier: itemIdentifier });

  if (error) console.error('Error saving item:', error);
}

export async function unsaveItem(userId: string, itemType: string, itemIdentifier: string) {
  const { error } = await supabase
    .from('user_saved_items')
    .delete()
    .match({ user_id: userId, item_type: itemType, item_identifier: itemIdentifier });

  if (error) console.error('Error unsaving item:', error);
}
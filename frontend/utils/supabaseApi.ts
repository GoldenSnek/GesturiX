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

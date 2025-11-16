import AsyncStorage from '@react-native-async-storage/async-storage';
import { supabase } from '../src/supabaseClient';
import { getCurrentUserId } from './supabaseApi';
import { phrases } from '../constants/phrases';

// Helper to get user-specific key
async function getUserKey(key: string): Promise<string | null> {
  const userId = await getCurrentUserId();
  if (!userId) return null;
  return `user_${userId}_${key}`;
}

export async function markPhraseCompleted(phraseId: string) {
  const userKey = await getUserKey(`phrase_${phraseId}_completed`);
  if (!userKey) return;
  
  await AsyncStorage.setItem(userKey, 'true');
  await syncCompletedPhrasesToSupabase();
  await updateStreakOnLessonComplete();
}

export async function isPhraseCompleted(phraseId: string): Promise<boolean> {
  const userKey = await getUserKey(`phrase_${phraseId}_completed`);
  if (!userKey) return false;
  
  const completed = await AsyncStorage.getItem(userKey);
  return completed === 'true';
}

export async function getCompletedPhrases(phraseIds: string[]): Promise<string[]> {
  const completed: string[] = [];
  for (const id of phraseIds) {
    if (await isPhraseCompleted(id)) completed.push(id);
  }
  return completed;
}

// Helper to sync local completed phrases to Supabase lessons_completed
export async function syncCompletedPhrasesToSupabase() {
  const userId = await getCurrentUserId();
  if (!userId) return;

  const phraseIds = phrases.map(p => p.id);
  const completed = await getCompletedPhrases(phraseIds);

  await supabase
    .from('user_statistics')
    .upsert([{ user_id: userId, lessons_completed: completed.length }], { onConflict: 'user_id' });
}

// Helper to update streak logic in Supabase upon lesson completion
export async function updateStreakOnLessonComplete() {
  const userId = await getCurrentUserId();
  if (!userId) return;

  const { data: stats } = await supabase
    .from('user_statistics')
    .select('days_streak, last_activity_date')
    .eq('user_id', userId)
    .single();

  let newStreak = 1;
  const today = new Date().toISOString().slice(0, 10);

  if (stats && stats.last_activity_date) {
    const lastDate = stats.last_activity_date;
    if (lastDate === today) {
      newStreak = stats.days_streak;
    } else {
      const yesterday = new Date();
      yesterday.setDate(yesterday.getDate() - 1);
      const yString = yesterday.toISOString().slice(0, 10);

      if (lastDate === yString) {
        newStreak = stats.days_streak + 1;
      } else {
        newStreak = 1;
      }
    }
  }

  await supabase
    .from('user_statistics')
    .upsert([{ user_id: userId, days_streak: newStreak, last_activity_date: today }], { onConflict: 'user_id' });
}

export async function updatePracticeTime(hoursToAdd: number) {
  const userId = await getCurrentUserId();
  if (!userId) return;

  const { data: stats } = await supabase
    .from('user_statistics')
    .select('practice_hours')
    .eq('user_id', userId)
    .single();

  const newHours = (stats?.practice_hours ?? 0) + hoursToAdd;

  await supabase
    .from('user_statistics')
    .upsert([{ user_id: userId, practice_hours: newHours }], { onConflict: 'user_id' });
}

// Clear local AsyncStorage for current user (call on logout)
export async function clearUserProgressData() {
  const userId = await getCurrentUserId();
  if (!userId) return;

  // Get all AsyncStorage keys for this user
  const allKeys = await AsyncStorage.getAllKeys();
  const userKeys = allKeys.filter(key => key.startsWith(`user_${userId}_`));
  
  if (userKeys.length > 0) {
    await AsyncStorage.multiRemove(userKeys);
  }
}

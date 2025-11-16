// File: utils/progressStorage.ts
import AsyncStorage from '@react-native-async-storage/async-storage';
import { supabase } from '../src/supabaseClient';
import { getCurrentUserId } from './supabaseApi';
import { phrases } from '../constants/phrases';
import { alphabetSigns } from '../constants/alphabetSigns';

// Helper to get user-specific key
async function getUserKey(key: string): Promise<string | null> {
  const userId = await getCurrentUserId();
  if (!userId) return null;
  return `user_${userId}_${key}`;
}

// ----- PHRASE PROGRESS -----
export async function markPhraseCompleted(phraseId: string) {
  const userKey = await getUserKey(`phrase_${phraseId}_completed`);
  if (!userKey) return;

  await AsyncStorage.setItem(userKey, 'true');
  await syncProgressToSupabase();
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

// ----- LETTER PROGRESS -----
export async function markLetterCompleted(letter: string) {
  const userKey = await getUserKey(`letter_${letter}_completed`);
  if (!userKey) return;

  await AsyncStorage.setItem(userKey, 'true');
  await syncProgressToSupabase();
  await updateStreakOnLessonComplete();
}

export async function isLetterCompleted(letter: string): Promise<boolean> {
  const userKey = await getUserKey(`letter_${letter}_completed`);
  if (!userKey) return false;

  const completed = await AsyncStorage.getItem(userKey);
  return completed === 'true';
}

export async function getCompletedLetters(letters: string[]): Promise<string[]> {
  const completed: string[] = [];
  for (const l of letters) {
    if (await isLetterCompleted(l)) completed.push(l);
  }
  return completed;
}

// ----- SYNC TO SUPABASE (both phrases & letters) -----
export async function syncProgressToSupabase() {
  const userId = await getCurrentUserId();
  if (!userId) return;

  const letterIds = alphabetSigns.map(l => l.letter);
  const phraseIds = phrases.map(p => p.id);
  const completedLetters = await getCompletedLetters(letterIds);
  const completedPhrases = await getCompletedPhrases(phraseIds);

  await supabase
    .from('user_statistics')
    .upsert(
      [{
        user_id: userId,
        lessons_completed: completedLetters.length + completedPhrases.length
      }],
      { onConflict: 'user_id' }
    );
}

// ✅ NEW: Reset only phrases and sync correctly
export async function resetPhraseProgress() {
  const userId = await getCurrentUserId();
  if (!userId) return;

  // Remove all phrase completion keys
  const allKeys = await AsyncStorage.getAllKeys();
  const phraseKeys = allKeys.filter(k => k.includes('_phrase_') && k.startsWith(`user_${userId}_`));
  if (phraseKeys.length > 0) {
    await AsyncStorage.multiRemove(phraseKeys);
  }

  // Remove last phrase tracker
  await AsyncStorage.removeItem('phrasescreen_last_phrase_id');

  // Sync the updated count (phrases = 0, but letters remain)
  await syncProgressToSupabase();
}

// ✅ NEW: Reset only letters and sync correctly
export async function resetLetterProgress() {
  const userId = await getCurrentUserId();
  if (!userId) return;

  // Remove all letter completion keys
  const allKeys = await AsyncStorage.getAllKeys();
  const letterKeys = allKeys.filter(k => k.includes('_letter_') && k.startsWith(`user_${userId}_`));
  if (letterKeys.length > 0) {
    await AsyncStorage.multiRemove(letterKeys);
  }

  // Remove last letter tracker
  await AsyncStorage.removeItem('letterscreen_last_letter');

  // Sync the updated count (letters = 0, but phrases remain)
  await syncProgressToSupabase();
}

// ----- STREAK LOGIC -----
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

// ----- PRACTICE HOURS TRACKING -----
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

// ----- CLEAR ALL LOCAL PROGRESS -----
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

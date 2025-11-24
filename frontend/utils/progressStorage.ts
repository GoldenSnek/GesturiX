// File: frontend/utils/progressStorage.ts
import AsyncStorage from '@react-native-async-storage/async-storage';
import { supabase } from '../src/supabaseClient';
import { getCurrentUserId } from './supabaseApi';
import { phrases } from '../constants/phrases';
import { alphabetSigns } from '../constants/alphabetSigns';
import { numbersData } from '../constants/numbers';

// --- HELPERS ---

/**
 * Efficiently retrieves the current user ID.
 */
async function getUserId(): Promise<string | null> {
  return await getCurrentUserId();
}

async function getUserKey(key: string): Promise<string | null> {
  const userId = await getUserId();
  if (!userId) return null;
  return `user_${userId}_${key}`;
}

/**
 * Batch fetches completion status for an array of IDs.
 * Uses AsyncStorage.multiGet for performance (1 roundtrip vs N roundtrips).
 */
async function getCompletedItems(ids: string[], typePrefix: string): Promise<string[]> {
  const userId = await getUserId();
  if (!userId) return [];

  // Construct keys: user_{userId}_{typePrefix}_{id}_completed
  const keys = ids.map(id => `user_${userId}_${typePrefix}_${id}_completed`);
  
  try {
    const stores = await AsyncStorage.multiGet(keys);
    const completed: string[] = [];
    
    // stores is an array of [key, value]
    stores.forEach((result, i) => {
      if (result[1] === 'true') {
        // We use the original 'ids' array index to match because multiGet preserves order
        completed.push(ids[i]);
      }
    });
    
    return completed;
  } catch (error) {
    console.error(`Error fetching ${typePrefix} progress:`, error);
    return [];
  }
}

// ----- PHRASE PROGRESS -----

export async function markPhraseCompleted(phraseId: string) {
  const userId = await getUserId();
  if (!userId) return;
  
  const key = `user_${userId}_phrase_${phraseId}_completed`;

  try {
    // 1. Instant Local Update
    await AsyncStorage.setItem(key, 'true');
    
    // 2. Background Cloud Sync (Don't await to unblock UI)
    syncProgressToSupabase().catch(err => console.error("Background Sync Error:", err));
    updateStreakOnLessonComplete().catch(err => console.error("Background Streak Error:", err));
  } catch (e) {
    console.error("Error marking phrase completed:", e);
  }
}

export async function isPhraseCompleted(phraseId: string): Promise<boolean> {
  const userKey = await getUserKey(`phrase_${phraseId}_completed`);
  if (!userKey) return false;
  const completed = await AsyncStorage.getItem(userKey);
  return completed === 'true';
}

export async function getCompletedPhrases(phraseIds: string[]): Promise<string[]> {
  return getCompletedItems(phraseIds, 'phrase');
}

// ----- LETTER PROGRESS -----

export async function markLetterCompleted(letter: string) {
  const userId = await getUserId();
  if (!userId) return;
  
  const key = `user_${userId}_letter_${letter}_completed`;

  try {
    await AsyncStorage.setItem(key, 'true');
    // Background Sync
    syncProgressToSupabase().catch(err => console.error("Background Sync Error:", err));
    updateStreakOnLessonComplete().catch(err => console.error("Background Streak Error:", err));
  } catch (e) {
    console.error("Error marking letter completed:", e);
  }
}

export async function isLetterCompleted(letter: string): Promise<boolean> {
  const userKey = await getUserKey(`letter_${letter}_completed`);
  if (!userKey) return false;
  const completed = await AsyncStorage.getItem(userKey);
  return completed === 'true';
}

export async function getCompletedLetters(letters: string[]): Promise<string[]> {
  return getCompletedItems(letters, 'letter');
}

// ----- NUMBER PROGRESS -----

export async function markNumberCompleted(numberVal: number) {
  const userId = await getUserId();
  if (!userId) return;
  
  const key = `user_${userId}_number_${numberVal}_completed`;

  try {
    await AsyncStorage.setItem(key, 'true');
    // Background Sync
    syncProgressToSupabase().catch(err => console.error("Background Sync Error:", err));
    updateStreakOnLessonComplete().catch(err => console.error("Background Streak Error:", err));
  } catch (e) {
    console.error("Error marking number completed:", e);
  }
}

export async function isNumberCompleted(numberVal: number): Promise<boolean> {
  const userKey = await getUserKey(`number_${numberVal}_completed`);
  if (!userKey) return false;
  const completed = await AsyncStorage.getItem(userKey);
  return completed === 'true';
}

export async function getCompletedNumbers(numbers: number[]): Promise<number[]> {
  const numberStringIds = numbers.map(n => n.toString());
  const completedStrings = await getCompletedItems(numberStringIds, 'number');
  return completedStrings.map(s => parseInt(s, 10));
}

// ----- SYNC TO SUPABASE (Optimized) -----

export async function syncProgressToSupabase() {
  const userId = await getUserId();
  if (!userId) return;

  // Parallelize local reads to calculate total
  const [completedLetters, completedPhrases, completedNumbers] = await Promise.all([
    getCompletedLetters(alphabetSigns.map(l => l.letter)),
    getCompletedPhrases(phrases.map(p => p.id)),
    getCompletedNumbers(numbersData.map(n => n.number))
  ]);

  const totalCompleted = completedLetters.length + completedPhrases.length + completedNumbers.length;

  // Perform upsert
  const { error } = await supabase
    .from('user_statistics')
    .upsert(
      [{
        user_id: userId,
        lessons_completed: totalCompleted,
        updated_at: new Date().toISOString()
      }],
      { onConflict: 'user_id' }
    );
    
  if (error) console.error("Supabase sync failed:", error.message);
}

// ----- RESET LOGIC (Optimized to use multiRemove) -----

export async function resetPhraseProgress() {
  const userId = await getUserId();
  if (!userId) return;

  try {
    const allKeys = await AsyncStorage.getAllKeys();
    const phraseKeys = allKeys.filter(k => k.includes('_phrase_') && k.startsWith(`user_${userId}_`));
    if (phraseKeys.length > 0) {
      await AsyncStorage.multiRemove(phraseKeys);
    }
    await AsyncStorage.removeItem('phrasescreen_last_phrase_id');
    syncProgressToSupabase();
  } catch (e) { console.error(e); }
}

export async function resetLetterProgress() {
  const userId = await getUserId();
  if (!userId) return;

  try {
    const allKeys = await AsyncStorage.getAllKeys();
    const letterKeys = allKeys.filter(k => k.includes('_letter_') && k.startsWith(`user_${userId}_`));
    if (letterKeys.length > 0) {
      await AsyncStorage.multiRemove(letterKeys);
    }
    await AsyncStorage.removeItem('letterscreen_last_letter');
    syncProgressToSupabase();
  } catch (e) { console.error(e); }
}

export async function resetNumberProgress() {
  const userId = await getUserId();
  if (!userId) return;

  try {
    const allKeys = await AsyncStorage.getAllKeys();
    const numberKeys = allKeys.filter(k => k.includes('_number_') && k.startsWith(`user_${userId}_`));
    if (numberKeys.length > 0) {
      await AsyncStorage.multiRemove(numberKeys);
    }
    await AsyncStorage.removeItem('numberscreen_last_number');
    syncProgressToSupabase();
  } catch (e) { console.error(e); }
}

// ----- STREAK LOGIC -----

export async function updateStreakOnLessonComplete() {
  const userId = await getUserId();
  if (!userId) return;

  try {
    const { data: stats, error } = await supabase
      .from('user_statistics')
      .select('days_streak, last_activity_date')
      .eq('user_id', userId)
      .single();

    if (error && error.code !== 'PGRST116') {
       console.error("Error fetching stats for streak:", error.message);
       return;
    }

    const today = new Date().toISOString().slice(0, 10);
    let newStreak = 1;
    let shouldUpdate = true;

    if (stats && stats.last_activity_date) {
      const lastDate = stats.last_activity_date;
      
      if (lastDate === today) {
        // Already updated today, save the DB call
        shouldUpdate = false; 
      } else {
        const yesterday = new Date();
        yesterday.setDate(yesterday.getDate() - 1);
        const yString = yesterday.toISOString().slice(0, 10);

        if (lastDate === yString) {
          newStreak = (stats.days_streak || 0) + 1;
        } else {
          newStreak = 1;
        }
      }
    }

    if (shouldUpdate) {
      await supabase
        .from('user_statistics')
        .upsert([{ 
            user_id: userId, 
            days_streak: newStreak, 
            last_activity_date: today 
        }], { onConflict: 'user_id' });
    }
  } catch (e) {
    console.error("Streak update exception:", e);
  }
}

export async function updatePracticeTime(hoursToAdd: number) {
  const userId = await getUserId();
  if (!userId) return;

  try {
    const { data: stats } = await supabase
      .from('user_statistics')
      .select('practice_hours')
      .eq('user_id', userId)
      .single();

    const newHours = (stats?.practice_hours ?? 0) + hoursToAdd;

    await supabase
      .from('user_statistics')
      .upsert([{ user_id: userId, practice_hours: newHours }], { onConflict: 'user_id' });
  } catch (e) { console.error(e); }
}

export async function clearUserProgressData() {
  const userId = await getUserId();
  if (!userId) return;
  try {
    const allKeys = await AsyncStorage.getAllKeys();
    const userKeys = allKeys.filter(key => key.startsWith(`user_${userId}_`));
    if (userKeys.length > 0) {
      await AsyncStorage.multiRemove(userKeys);
    }
  } catch (e) { console.error(e); }
}
import AsyncStorage from '@react-native-async-storage/async-storage';

export async function markPhraseCompleted(phraseId: string) {
  await AsyncStorage.setItem(`phrase_${phraseId}_completed`, 'true');
}

export async function isPhraseCompleted(phraseId: string): Promise<boolean> {
  const completed = await AsyncStorage.getItem(`phrase_${phraseId}_completed`);
  return completed === 'true';
}

export async function getCompletedPhrases(phraseIds: string[]): Promise<string[]> {
  const completed: string[] = [];
  for (const id of phraseIds) {
    if (await isPhraseCompleted(id)) completed.push(id);
  }
  return completed;
}

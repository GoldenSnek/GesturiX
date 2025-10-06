import { Stack } from 'expo-router';

const LearnStack = () => {
  return (
    <Stack screenOptions={{ headerShown: false, animation: 'fade' }}>
      <Stack.Screen name="index" options={{ headerShown: false }} />
      <Stack.Screen name="Letters" options={{ title: 'Letters' }} />
      <Stack.Screen name="Numbers" options={{ title: 'Numbers' }} />
      <Stack.Screen name="Phrases" options={{ title: 'Phrases' }} />
    </Stack>
  );
}; 

export default LearnStack;
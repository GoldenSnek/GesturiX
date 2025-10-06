import { Stack } from 'expo-router';

const LearnStack = () => {
  return (
    <Stack screenOptions={{ headerShown: false, animation: 'fade' }}>
      <Stack.Screen name="index" options={{ headerShown: false }} />
      <Stack.Screen name="letters" options={{ title: 'Letters' }} />
      <Stack.Screen name="numbers" options={{ title: 'Numbers' }} />
      <Stack.Screen name="phrases" options={{ title: 'Phrases' }} />
    </Stack>
  );
}; 

export default LearnStack;
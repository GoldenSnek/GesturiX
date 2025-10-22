// File: app/compose.tsx
import React, { useRef } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  PanResponder,
} from 'react-native';
import { MaterialIcons, MaterialCommunityIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useTheme } from '../../src/ThemeContext'; // ✅ Use your global theme
import AppHeader from '../../components/AppHeader';

const Compose = () => {
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const { isDark } = useTheme(); // ✅ Access current theme

  const panResponder = useRef(
    PanResponder.create({
      onMoveShouldSetPanResponder: (_, gestureState) =>
        Math.abs(gestureState.dx) > 10,
      onPanResponderRelease: (_, gestureState) => {
        if (gestureState.dx < -30) {
          router.push('/learn');
        } else if (gestureState.dx > 30) {
          router.push('/translate');
        }
      },
    })
  ).current;

  return (
    <View
      {...panResponder.panHandlers}
      className={`flex-1 ${isDark ? 'bg-darkbg' : 'bg-secondary'}`}
      style={{ paddingTop: insets.top }}
    >
      <AppHeader />

      <ScrollView
        className="flex-1 p-4"
        contentContainerStyle={{ paddingBottom: 150 }}
      >
        {/* 🗣️ Text Input Section */}
        <View
          className={`rounded-2xl shadow-md p-5 mb-8 border ${
            isDark
              ? 'bg-darksurface border-accent'
              : 'bg-white border-accent'
          }`}
        >
          <View className="flex-row justify-between items-center mb-4">
            <Text
              className={`${isDark ? 'text-secondary' : 'text-primary'} text-xl tracking-wide`}
              style={{ fontFamily: 'Fredoka-SemiBold' }}
            >
              Type or Speak
            </Text>

            <TouchableOpacity className="bg-accent rounded-full p-3 shadow-sm active:opacity-80">
              <MaterialCommunityIcons
                name="microphone"
                size={22}
                color="white"
              />
            </TouchableOpacity>
          </View>

          <TextInput
            className={`h-28 p-4 text-base rounded-xl border ${
              isDark
                ? 'bg-darkhover border-darkhover text-secondary'
                : 'bg-secondary border-neutral text-primary'
            }`}
            placeholder="Type something to see its sign..."
            placeholderTextColor={isDark ? '#A8A8A8' : '#888'}
            multiline
            style={{ fontFamily: 'Montserrat-SemiBold' }}
          />
        </View>

        {/* 🎥 Sign Language Video Section */}
        <Text
          className={`${isDark ? 'text-secondary' : 'text-primary'} text-lg mb-3`}
          style={{ fontFamily: 'Audiowide-Regular', letterSpacing: 0.5 }}
        >
          Sign Language Video
        </Text>

        <View
          className={`w-full aspect-[4/3] rounded-3xl overflow-hidden mb-7 relative ${
            isDark ? 'bg-darksurface' : 'bg-primary'
          }`}
        >
          <View className="flex-1 justify-center items-center px-5">
            <MaterialIcons
              name="videocam"
              size={80}
              color={isDark ? '#777' : '#A8A8A8'}
            />
            <Text
              className={`${isDark ? 'text-neutral' : 'text-neutral'} text-lg mt-4 mb-2`}
              style={{ fontFamily: 'Montserrat-SemiBold' }}
            >
              Video Placeholder
            </Text>
            <Text
              className="text-sm text-neutral/50 text-center leading-5"
              style={{ fontFamily: 'Inter-Regular' }}
            >
              The corresponding sign will appear here
            </Text>
          </View>

          {/* 🎬 Frame Corners */}
          <View className="absolute top-4 left-4 w-6 h-6 border-t-[3px] border-l-[3px] border-accent" />
          <View className="absolute top-4 right-4 w-6 h-6 border-t-[3px] border-r-[3px] border-accent" />
          <View className="absolute bottom-4 left-4 w-6 h-6 border-b-[3px] border-l-[3px] border-accent" />
          <View className="absolute bottom-4 right-4 w-6 h-6 border-b-[3px] border-r-[3px] border-accent" />
        </View>

        {/* 💬 Quick Phrases Section */}
        <Text
          className={`${isDark ? 'text-secondary' : 'text-primary'} text-lg mb-3`}
          style={{ fontFamily: 'Fredoka-SemiBold' }}
        >
          Quick Phrases
        </Text>

        <View className="flex-row flex-wrap justify-between">
          {['Hello', 'Thank You', 'How are you?', 'I love you', 'Good morning'].map(
            (phrase) => (
              <TouchableOpacity
                key={phrase}
                className={`rounded-full px-5 py-2 my-1 shadow-sm active:opacity-80 ${
                  isDark ? 'bg-accent' : 'bg-highlight'
                }`}
              >
                <Text
                  className={`${isDark ? 'text-darkbg' : 'text-primary'} text-base`}
                  style={{ fontFamily: 'Audiowide-Regular' }}
                >
                  {phrase}
                </Text>
              </TouchableOpacity>
            )
          )}
        </View>
      </ScrollView>
    </View>
  );
};

export default Compose;
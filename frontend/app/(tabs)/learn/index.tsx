import React, { useRef, useState, useMemo, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  PanResponder,
  ImageBackground,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useTheme } from '../../../src/ThemeContext';
import { LinearGradient } from 'expo-linear-gradient';
import Animated, { 
  useSharedValue, 
  useAnimatedStyle, 
  withRepeat, 
  withTiming, 
  withSequence,
  FadeInDown,
  Easing,
  interpolate,
  useDerivedValue
} from 'react-native-reanimated';

import { phrases } from '../../../constants/phrases';
import { alphabetSigns } from '../../../constants/alphabetSigns';
import { numbersData } from '../../../constants/numbers';
import { getCompletedPhrases, getCompletedLetters, getCompletedNumbers } from '../../../utils/progressStorage';
import { useFocusEffect } from '@react-navigation/native';

import { fetchUserStatistics, getCurrentUserId } from '../../../utils/supabaseApi';

const totalPhrases = phrases.length;
const totalLetters = alphabetSigns.length;
const totalNumbers = numbersData.length;

// Create an animated component for the icon
const AnimatedIcon = Animated.createAnimatedComponent(MaterialIcons);

const ProgressCard = ({ item, index, isDark }: { item: any, index: number, isDark: boolean }) => {
  // Shared values for different animations
  const scale = useSharedValue(1);
  const rotation = useSharedValue(0);
  const translateY = useSharedValue(0);

  useEffect(() => {
    // 1. Streak Animation: Pulse (Fire)
    if (item.isStreak && parseInt(item.rawValue) > 0) {
      scale.value = withRepeat(
        withSequence(
          withTiming(1.2, { duration: 700, easing: Easing.inOut(Easing.ease) }),
          withTiming(1, { duration: 700, easing: Easing.inOut(Easing.ease) })
        ),
        -1, // Infinite
        true // Reverse
      );
    }

    // 2. Clock Animation: Ticking / Rotation
    if (item.icon === 'schedule') {
      rotation.value = withRepeat(
        withTiming(360, { duration: 6000, easing: Easing.linear }),
        -1, // Infinite loop
        false // Do not reverse
      );
    }

    // 3. Hat Animation: Bounce / Toss (Made slightly more pronounced for UX)
    if (item.icon === 'school') {
      translateY.value = withRepeat(
        withSequence(
          withTiming(-5, { duration: 600, easing: Easing.out(Easing.quad) }), // Jump up higher
          withTiming(0, { duration: 600, easing: Easing.in(Easing.quad) }),   // Fall down
          withTiming(0, { duration: 1000 }) // Pause on ground
        ),
        -1,
        false
      );
    }
  }, [item.isStreak, item.rawValue, item.icon]);

  const animatedIconStyle = useAnimatedStyle(() => {
    return {
      transform: [
        { scale: scale.value },
        { rotate: `${rotation.value}deg` },
        { translateY: translateY.value }
      ],
    };
  });

  // Gradient colors based on theme and card type
  const getGradientColors = () => {
    if (item.isStreak) return ['#FF6B00', '#FF8E2D']; // Orange fire
    if (isDark) return ['#2A2A2A', '#1F1F1F'];
    return ['#FFFFFF', '#F0F0F0'];
  };

  const textColor = item.isStreak ? '#FFF' : (isDark ? '#E2E8F0' : '#333');
  const subTextColor = item.isStreak ? 'rgba(255,255,255,0.8)' : (isDark ? '#94A3B8' : '#666');
  const iconColor = item.isStreak ? '#FFF' : '#FF6B00';

  return (
    <Animated.View 
      entering={FadeInDown.delay(index * 150).springify().damping(12)}
      className="mr-3"
      style={{
        shadowColor: item.isStreak ? '#FF6B00' : '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: item.isStreak ? 0.4 : 0.1,
        shadowRadius: item.isStreak ? 8 : 3,
        elevation: item.isStreak ? 8 : 3,
      }}
    >
      <LinearGradient
        colors={getGradientColors() as any}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        className={`rounded-2xl p-4 items-center justify-center min-w-[8rem] h-24 border ${
          item.isStreak ? 'border-transparent' : 'border-accent'
        }`}
        style={{ borderRadius: 16 }}
      >
        <AnimatedIcon 
          name={item.icon} 
          size={32} 
          color={iconColor} 
          style={[{ marginBottom: 6 }, animatedIconStyle]} 
        />
        <Text
          className="text-xl font-bold text-center"
          style={{ fontFamily: 'Fredoka-SemiBold', color: textColor }}
        >
          {item.value}
        </Text>
        <Text
          className="text-xs text-center mt-1"
          style={{ fontFamily: 'Montserrat-SemiBold', color: subTextColor }}
        >
          {item.label}
        </Text>
      </LinearGradient>
    </Animated.View>
  );
};

const Learn = () => {
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const { isDark } = useTheme();

  const bgColorClass = isDark ? 'bg-darkbg' : 'bg-secondary';

  const [phrasesCompleted, setPhrasesCompleted] = useState(0);
  const [lettersCompleted, setLettersCompleted] = useState(0);
  const [numbersCompleted, setNumbersCompleted] = useState(0);

  const [userStats, setUserStats] = useState({
    lessons_completed: 0,
    days_streak: 0,
    practice_hours: 0,
  });

  // State to force re-render of animations on focus
  const [animationKey, setAnimationKey] = useState(0);
  const isFirstRender = useRef(true);

  const quickActionsData = [
    { id: 'leaderboard', title: 'Leaderboard', subtitle: 'Top learners', icon: 'leaderboard', route: '/(tabs)/learn/leaderboard' },
    { id: 'dictionary', title: 'Dictionary', subtitle: 'Browse all signs', icon: 'menu-book', route: '/(tabs)/learn/dictionary' },
    { id: 'video', title: 'Video Lessons', subtitle: 'Watch and Learn', icon: 'ondemand-video', route: '/(tabs)/learn/videos' }, 
    { id: 'saved', title: 'Saved Signs', subtitle: 'Your favorites', icon: 'bookmark-outline', route: '/(tabs)/learn/saved' },
  ];

  useFocusEffect(
    React.useCallback(() => {
      let isActive = true;

      // Logic to trigger animation restart on return visits
      if (isFirstRender.current) {
        isFirstRender.current = false;
      } else {
        setAnimationKey(prev => prev + 1);
      }

      const loadData = async () => {
        try {
          const progressPromise = Promise.all([
            getCompletedPhrases(phrases.map(p => p.id)),
            getCompletedLetters(alphabetSigns.map(l => l.letter)),
            getCompletedNumbers(numbersData.map(n => n.number))
          ]);

          const statsPromise = (async () => {
            const userId = await getCurrentUserId();
            if (typeof userId === 'string') {
              return await fetchUserStatistics(userId);
            }
            return null;
          })();

          const [[donePhrases, doneLetters, doneNumbers], stats] = await Promise.all([
            progressPromise,
            statsPromise
          ]);

          if (isActive) {
            setPhrasesCompleted(donePhrases.length);
            setLettersCompleted(doneLetters.length);
            setNumbersCompleted(doneNumbers.length);
            if (stats) setUserStats(stats);
          }
        } catch (error) {
          console.error("Failed to load progress:", error);
        }
      };

      loadData();

      return () => { isActive = false; };
    }, [])
  );

  const getFormattedPracticeTime = (hours: number) => {
    if (hours < (1 / 60)) {
       return "0 mins";
    }

    if (hours < 1) {
      const mins = Math.round(hours * 60);
      return `${mins} ${mins === 1 ? 'min' : 'mins'}`;
    } else if (hours < 100) {
      return `${hours.toFixed(1)} hrs`;
    } else {
      return `${Math.round(hours).toLocaleString()} hrs`;
    }
  };

  const progressData = [
    { 
      label: 'Lessons Completed', 
      value: `${userStats.lessons_completed.toLocaleString()}`,
      rawValue: userStats.lessons_completed,
      icon: 'school'
    },
    { 
      label: 'Streak', 
      value: `${userStats.days_streak.toLocaleString()} ${userStats.days_streak === 1 ? 'day' : 'days'}`,
      rawValue: userStats.days_streak,
      icon: 'whatshot',
      isStreak: true
    },
    { 
      label: 'Learning Time', 
      value: getFormattedPracticeTime(userStats.practice_hours),
      rawValue: userStats.practice_hours,
      icon: 'schedule'
    },
  ];

  const categoriesData = useMemo(() => [
    {
      id: 'letters',
      title: 'Letters',
      subtitle: 'Learn the Alphabet',
      icon: 'text-fields',
      completed: lettersCompleted,
      total: totalLetters,
      progress: totalLetters > 0 ? lettersCompleted / totalLetters : 0,
    },
    {
      id: 'numbers',
      title: 'Numbers',
      subtitle: 'Count in sign language',
      icon: 'format-list-numbered',
      completed: numbersCompleted,
      total: totalNumbers,
      progress: totalNumbers > 0 ? numbersCompleted / totalNumbers : 0,
    },
    {
      id: 'phrases',
      title: 'Phrases',
      subtitle: 'Common expressions',
      icon: 'record-voice-over',
      completed: phrasesCompleted,
      total: totalPhrases,
      progress: totalPhrases > 0 ? phrasesCompleted / totalPhrases : 0,
    },
  ], [lettersCompleted, phrasesCompleted, numbersCompleted]);

  const panResponder = useRef(
    PanResponder.create({
      onMoveShouldSetPanResponder: (_, gestureState) =>
        Math.abs(gestureState.dx) > 10,
      onPanResponderRelease: (_, gestureState) => {
        if (gestureState.dx < -30) {
            router.push('/quiz');
        } else if (gestureState.dx > 30) {
            router.push('/translate');
        }
      },
    })
  ).current;

  const handleQuickAction = (route: string | null) => {
    if (route) {
      router.push(route as any);
    }
  };

  return (
    <View className={`flex-1 ${bgColorClass}`}>
      <ImageBackground
        source={require('../../../assets/images/MainBG.png')}
        className="flex-1"
        resizeMode="cover"
      >
        <View
          {...panResponder.panHandlers}
          className="flex-1"
          style={{ paddingTop: insets.top }}
        >
          <ScrollView
            className="flex-1 px-4 py-6"
            contentContainerStyle={{ paddingBottom: 150 }}
          >
            {/* Progress Section */}
            <Text
              className={`text-2xl mb-4 ${
                isDark ? 'text-secondary' : 'text-primary'
              }`}
              style={{ fontFamily: 'Audiowide-Regular' }}
            >
              Your Progress
            </Text>
            
            <ScrollView 
              horizontal 
              showsHorizontalScrollIndicator={false} 
              className="mb-6"
              contentContainerStyle={{ paddingRight: 20 }}
            >
              {progressData.map((item, index) => (
                <ProgressCard 
                  key={`progress-${index}-${animationKey}`} 
                  item={item} 
                  index={index} 
                  isDark={isDark} 
                />
              ))}
            </ScrollView>

            {/* Category Section */}
            <Text
              className={`text-2xl mb-4 ${
                isDark ? 'text-secondary' : 'text-primary'
              }`}
              style={{ fontFamily: 'Audiowide-Regular' }}
            >
              Choose Category
            </Text>

            {categoriesData.map((category, index) => (
              <Animated.View
                key={`cat-${category.id}-${animationKey}`}
                entering={FadeInDown.delay(index * 100 + 300).springify().damping(12)}
              >
                <TouchableOpacity
                  className={`rounded-2xl p-5 mb-4 shadow-md flex-row items-center border-2 border-accent ${
                    isDark ? 'bg-darksurface' : 'bg-highlight'
                  }`}
                  onPress={() => router.push(`/(tabs)/learn/${category.id}` as any)}
                >
                  <View
                    className={`w-16 h-16 rounded-full items-center justify-center mr-4 ${
                      isDark ? 'bg-darkhover' : 'bg-white/30'
                    }`}
                  >
                    <MaterialIcons
                      name={category.icon as any}
                      size={36}
                      color="#FF6B00"
                    />
                  </View>

                  <View className="flex-1">
                    <Text
                      className={`text-2xl ${
                        isDark ? 'text-secondary' : 'text-darkbg'
                      }`}
                      style={{ fontFamily: 'Fredoka-SemiBold' }}
                    >
                      {category.title}
                    </Text>
                    <Text
                      className={`text-sm mb-2 ${
                        isDark ? 'text-neutral' : 'text-darkbg'
                      }`}
                      style={{ fontFamily: 'Montserrat-SemiBold' }}
                    >
                      {category.subtitle}
                    </Text>
                    <View
                      className={`w-full h-2 rounded-full ${
                        isDark ? 'bg-darkhover' : 'bg-white/30'
                      }`}
                    >
                      <View
                        style={{ width: `${category.progress * 100}%` }}
                        className="h-full bg-accent rounded-full"
                      />
                    </View>
                    <Text
                      className={`text-xs mt-1 ${
                        isDark ? 'text-neutral' : 'text-darkbg'
                      }`}
                      style={{ fontFamily: 'Montserrat-SemiBold' }}
                    >
                      {category.completed} of {category.total} completed
                    </Text>
                  </View>
                </TouchableOpacity>
              </Animated.View>
            ))}

            {/* Quick Actions */}
            <Text
              className={`text-2xl mb-4 ${
                isDark ? 'text-secondary' : 'text-primary'
              }`}
              style={{ fontFamily: 'Audiowide-Regular' }}
            >
              Quick Actions
            </Text>

            <View className="flex-row flex-wrap justify-between">
              {quickActionsData.map((action, index) => (
                <Animated.View
                  key={`action-${action.id}-${animationKey}`}
                  entering={FadeInDown.delay(index * 100 + 600).springify().damping(12)}
                  className="w-[48%] mb-4"
                >
                  <TouchableOpacity
                    onPress={() => handleQuickAction(action.route)}
                    className={`rounded-2xl p-4 items-center justify-center h-32 shadow-sm border border-accent ${
                      isDark ? 'bg-darksurface' : 'bg-secondary'
                    }`}
                  >
                    <MaterialIcons name={action.icon as any} size={30} color="#FF6B00" />
                    <Text
                      className={`text-base text-center mt-2 ${
                        isDark ? 'text-highlight' : 'text-primary'
                      }`}
                      style={{ fontFamily: 'Fredoka-SemiBold' }}
                    >
                      {action.title}
                    </Text>
                    <Text
                      className={`text-xs text-center mt-1 ${
                        isDark ? 'text-neutral' : 'text-neutral'
                      }`}
                      style={{ fontFamily: 'Montserrat-SemiBold' }}
                    >
                      {action.subtitle}
                    </Text>
                  </TouchableOpacity>
                </Animated.View>
              ))}
            </View>
          </ScrollView>
        </View>
      </ImageBackground>
    </View>
  );
};

export default Learn;
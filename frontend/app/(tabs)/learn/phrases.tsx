import React, { useState, useEffect, useRef, useMemo } from 'react';
import { View, Text, TouchableOpacity, ScrollView } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';
import AppHeaderLearn from '../../../components/AppHeaderLearn';
import { Video, ResizeMode } from 'expo-av';
import { phrases } from '../../../constants/phrases';
import { markPhraseCompleted, getCompletedPhrases } from '../../../utils/progressStorage';
import { useTheme } from '../../../src/ThemeContext';
import AsyncStorage from '@react-native-async-storage/async-storage';

const STORAGE_LAST_PHRASE = 'phrasescreen_last_phrase_id';

const CATEGORIES = [
  { key: 'greetings', label: 'Greetings' },
  { key: 'courtesy', label: 'Courtesy' },
  { key: 'questions', label: 'Questions' },
];

async function resetAllPhraseProgress() {
  const keys = phrases.map((p) => `phrase_${p.id}_completed`);
  await AsyncStorage.multiRemove(keys);
  await AsyncStorage.removeItem(STORAGE_LAST_PHRASE);
}

export default function PhraseLearnScreen() {
  const insets = useSafeAreaInsets();
  const { isDark } = useTheme();
  const scrollRef = useRef<ScrollView>(null);

  const [doneIds, setDoneIds] = useState<string[]>([]);
  const [activeCategory, setActiveCategory] = useState(CATEGORIES[0].key);

  const phrasesForCategory = useMemo(
    () => phrases.filter(p => p.category === activeCategory), [activeCategory]
  );

  const [selectedPhrase, setSelectedPhrase] = useState(phrasesForCategory[0]);
  const [completed, setCompleted] = useState(false);

  // scrollToStart triggers once when category changes
  useEffect(() => {
    if (phrasesForCategory.length > 0) {
      setSelectedPhrase(phrasesForCategory[0]);
      setTimeout(() => {
        if (scrollRef.current) {
          scrollRef.current.scrollTo({ x: 0, animated: true });
        }
      }, 80);
    }
   
  }, [activeCategory, phrasesForCategory]);

  useEffect(() => {
    (async () => {
      const lastId = await AsyncStorage.getItem(STORAGE_LAST_PHRASE);
      const match = phrases.find(p => p.id === lastId);
      if (match) {
        setActiveCategory(match.category);
        setSelectedPhrase(match);
      } else {
        setActiveCategory(CATEGORIES[0].key);
        setSelectedPhrase(phrases.filter(p => p.category === CATEGORIES[0].key)[0]);
      }
      const done = await getCompletedPhrases(phrases.map(p => p.id));
      setDoneIds(done);
    })();
  }, []);

  useEffect(() => {
    if (selectedPhrase?.id) {
      AsyncStorage.setItem(STORAGE_LAST_PHRASE, selectedPhrase.id);
    }
  }, [selectedPhrase]);

  useEffect(() => {
    setCompleted(selectedPhrase && doneIds.includes(selectedPhrase.id));
  }, [selectedPhrase, doneIds]);

  // Progression logic
  const handleComplete = async () => {
    const allPhrasesInCat = phrases.filter(p => p.category === activeCategory);
    const currentIdx = allPhrasesInCat.findIndex(p => p.id === selectedPhrase.id);
    await markPhraseCompleted(selectedPhrase.id);
    const done = await getCompletedPhrases(phrases.map(p => p.id));
    setDoneIds(done);
    setCompleted(true);

    if (currentIdx < allPhrasesInCat.length - 1) {
      setTimeout(() => {
        setSelectedPhrase(allPhrasesInCat[currentIdx + 1]);
        if (scrollRef.current) {
          if (currentIdx + 1 === allPhrasesInCat.length - 1) {
            scrollRef.current.scrollToEnd({ animated: true });
          } else if (currentIdx + 1 > 0 && currentIdx + 1 < allPhrasesInCat.length - 1) {
            const itemWidth = 90 + 18 * 2;
            const gap = 12;
            const scrollToX = (itemWidth + gap) * (currentIdx + 1) - 1;
            scrollRef.current.scrollTo({ x: scrollToX, animated: true });
          }
        }
      }, 200);

    } else {
      // Finished this category: move to FIRST phrase in next category
      const currentCatIdx = CATEGORIES.findIndex(c => c.key === activeCategory);
      if (currentCatIdx < CATEGORIES.length - 1) {
        setTimeout(() => {
          setActiveCategory(CATEGORIES[currentCatIdx + 1].key);
        }, 200);
      }
    }
  };

  const handleResetProgress = async () => {
    await resetAllPhraseProgress();
    const done = await getCompletedPhrases(phrases.map(p => p.id));
    setDoneIds(done);
    setCompleted(false);
    setSelectedPhrase(phrases.filter(p => p.category === activeCategory)[0]);
  };

  return (
    <View
      className={`flex-1 ${isDark ? 'bg-darkbg' : 'bg-secondary'}`}
      style={{ paddingTop: insets.top }}
    >
      <AppHeaderLearn
        title="Learn Phrases"
        completedCount={doneIds.length}
        totalCount={phrases.length}
        onResetProgress={handleResetProgress}
      />

      <ScrollView className="flex-1 p-4" contentContainerStyle={{ paddingBottom: 150 }}>
        {/* Category Tabs Pill/Oval */}
<View
  style={{
    flexDirection: 'row',
    justifyContent: 'space-between',
    backgroundColor: isDark ? '#23201C' : '#f9f6f0',
    borderRadius: 28,
    padding: 7,
    marginBottom: 20,
    marginHorizontal: 12,
    borderWidth: 1,
    borderColor: isDark ? '#B5B1A2' : '#E5DDD4',
    shadowColor: '#000',
    shadowOpacity: 0.09,
    shadowOffset: { width: 0, height: 2 },
    shadowRadius: 4,
    elevation: isDark ? 2 : 1,
  }}
>
  {CATEGORIES.map((cat, i) => (
    <React.Fragment key={cat.key}>
      <TouchableOpacity
        onPress={() => setActiveCategory(cat.key)}
        activeOpacity={0.9}
        style={{
          flex: 1,
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius: 18,
          paddingVertical: 1,
          backgroundColor: activeCategory === cat.key
            ? (isDark ? '#F7CD84' : '#FFD3A6')
            : (isDark ? '#292822' : '#FAF3E7'),
          marginHorizontal: 5,
          transform: [{ scale: activeCategory === cat.key ? 1.05 : 1.0 }],
          shadowColor: activeCategory === cat.key ? '#FF6B00' : undefined,
          shadowOpacity: activeCategory === cat.key ? 0.14 : 0,
          shadowRadius: activeCategory === cat.key ? 6 : 0,
          elevation: activeCategory === cat.key ? 2 : 0,
          borderWidth: activeCategory === cat.key ? 2 : 0,
          borderColor: activeCategory === cat.key
            ? (isDark ? '#FF6B00' : '#FF6B00')
            : 'transparent',
        }}
      >
        <Text style={{
          fontFamily: 'Fredoka-SemiBold',
          fontSize: 15,
          color: activeCategory === cat.key
            ? (isDark ? '#A85600' : '#FF6B00')
            : (isDark ? '#FFD1A5' : '#A57D51'),
          letterSpacing: 1.1,
        }}>
          {cat.label}
        </Text>
      </TouchableOpacity>
      {i < CATEGORIES.length - 1 && (
        <View style={{
          width: 2,
          alignSelf: 'stretch',
          backgroundColor: isDark ? '#B5B1A2' : '#E5DDD4',
          marginVertical: 6,
          borderRadius: 99,
        }} />
      )}
    </React.Fragment>
  ))}
</View>

        {/* Horizontal Phrase Selector */}
        <ScrollView
          ref={scrollRef}
          horizontal
          showsHorizontalScrollIndicator={false}
          style={{ marginBottom: 16 }}
          contentContainerStyle={{ gap: 12, paddingRight: 8 }}
        >
          {phrasesForCategory.map((phrase, idx) => {
            const prevCompleted = idx === 0 || doneIds.includes(phrasesForCategory[idx - 1].id);
            const isCompleted = doneIds.includes(phrase.id);
            const isSelected = selectedPhrase.id === phrase.id;

            let backgroundColor, borderColor, textColor;
            if (isCompleted) {
              backgroundColor = isDark ? '#1e1e1e' : '#faf3ec';
              borderColor = '#FF6B00';
              textColor = '#FF6B00';
            } else if (!prevCompleted) {
              backgroundColor = isDark ? '#222' : '#EFEFEF';
              borderColor = isDark ? '#414141' : '#BDBDBD';
              textColor = isDark ? '#A0A0A0' : '#BDBDBD';
            } else if (isSelected) {
              backgroundColor = isDark ? '#FFAB7B' : '#FF6B00';
              borderColor = '#FF6B00';
              textColor = isDark ? '#1a1a1a' : '#fff';
            } else {
              backgroundColor = isDark ? '#242424' : '#FFF3E6';
              borderColor = isDark ? '#333' : '#ecd7c3';
              textColor = isDark ? '#ccc' : '#FF6B00';
            }
            return (
              <TouchableOpacity
                key={phrase.id}
                onPress={() => {
                  if (prevCompleted) setSelectedPhrase(phrase);
                }}
                activeOpacity={prevCompleted ? 0.8 : 1}
                disabled={!prevCompleted}
                style={{
                  backgroundColor,
                  paddingVertical: 12,
                  paddingHorizontal: 18,
                  borderRadius: 13,
                  borderWidth: (isCompleted || isSelected) ? 2 : 1,
                  borderColor,
                  minWidth: 90,
                  alignItems: 'center',
                  justifyContent: 'center',
                  opacity: prevCompleted ? 1 : 0.6,
                  position: 'relative',
                }}
              >
                <Text
                  style={{
                    fontFamily: 'Fredoka-SemiBold',
                    fontSize: 16,
                    color: textColor,
                  }}
                >
                  {phrase.text}
                </Text>
                {isCompleted && (
                  <View style={{ position: 'absolute', top: 5, right: 8 }}>
                    <MaterialIcons name="check-circle" size={18} color="#FF6B00" />
                  </View>
                )}
              </TouchableOpacity>
            );
          })}
        </ScrollView>
        {/* Selected Phrase View */}
        <Text style={{
          fontSize: 24,
          fontFamily: 'Fredoka-SemiBold',
          color: isDark ? '#fff' : '#1A1A1A',
          marginBottom: 4,
        }}>
          {selectedPhrase.text}
        </Text>
        <View
          style={{
            width: '100%',
            height: 230,
            borderRadius: 20,
            backgroundColor: isDark ? '#222' : '#fffcfa',
            borderWidth: 2,
            borderColor: isDark ? '#FFB366' : '#FF6B00', // a vibrant accent border
            shadowColor: isDark ? '#FFB366' : '#FF6B00',
            shadowOffset: { width: 0, height: 3 },
            shadowOpacity: 0.12,
            shadowRadius: 8,
            elevation: 4,
            marginBottom: 16,
            alignItems: 'center',
            justifyContent: 'center',
            overflow: 'hidden',
          }}
        >
          <Video
            source={selectedPhrase.videoUrl}
            rate={1.0}
            volume={1.0}
            isMuted={false}
            resizeMode={ResizeMode.CONTAIN}
            shouldPlay={false}
            useNativeControls
            style={{
              width: '100%',
              height: 220,
              borderRadius: 18,
              backgroundColor: isDark ? '#181818' : '#f5f5f5',
            }}
          />
        </View>

        {/* Tips Section */}
        <Text
          style={{
            marginVertical: 8,
            paddingHorizontal: 4,
            flexDirection: 'row',
            flexWrap: 'wrap',
          }}
        >
          <Text
            style={{
              fontWeight: 'bold',
              fontFamily: 'Montserrat-Bold',
              fontSize: 14,
              color: isDark ? '#FFA500' : '#FF6B00',
            }}
          >
            Tips:
          </Text>
          <Text
            style={{
              fontFamily: 'Montserrat-SemiBold',
              color: isDark ? '#ccc' : '#333',
              fontSize: 13,
            }}
          >
            {' '}{selectedPhrase.tips}
          </Text>
        </Text>
        <TouchableOpacity
          onPress={handleComplete}
          disabled={completed}
          style={{
            backgroundColor: completed ? 'gray' : '#FF6B00',
            padding: 16,
            borderRadius: 24,
            marginTop: 10,
            alignItems: 'center'
          }}
        >
          <Text style={{
            color: 'white',
            fontFamily: 'Fredoka-SemiBold',
            fontSize: 17
          }}>
            {completed ? 'Completed!' : 'Mark as Completed'}
          </Text>
        </TouchableOpacity>
      </ScrollView>
    </View>
  );
}

import React, { useState } from 'react';
import { View, Text, TouchableOpacity, ScrollView, StyleSheet } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';
import AppHeaderLearn from '../../../components/AppHeaderLearn';
import { useTheme } from '../../../src/ThemeContext';

// ✅ Define category key types
type CategoryKey = 'greetings' | 'courtesy' | 'questions';

// ✅ Define phrase item structure
interface PhraseItem {
  phrase: string;
  completed: boolean;
}

// ✅ Phrase data
const phraseData: Record<CategoryKey, PhraseItem[]> = {
  greetings: [
    { phrase: 'Hello', completed: true },
    { phrase: 'Good Morning', completed: true },
    { phrase: 'Good Afternoon', completed: true },
    { phrase: 'Good Evening', completed: false },
    { phrase: 'Goodbye', completed: false },
    { phrase: 'See you later', completed: false },
    { phrase: 'Nice to meet you', completed: false },
  ],
  courtesy: [
    { phrase: 'Thank you', completed: false },
    { phrase: "You're welcome", completed: false },
    { phrase: 'Please', completed: false },
    { phrase: 'Excuse me', completed: false },
  ],
  questions: [
    { phrase: 'How are you?', completed: false },
    { phrase: 'What is your name?', completed: false },
    { phrase: 'Where are you from?', completed: false },
    { phrase: 'What is your favorite color?', completed: false },
  ],
};

const Phrases = () => {
  const insets = useSafeAreaInsets();
  const { isDark } = useTheme();

  const [activeCategory, setActiveCategory] = useState<CategoryKey>('greetings');
  const [activePhrase, setActivePhrase] = useState('Hello');

  const allPhrasesCount = Object.values(phraseData).flat().length;
  const completedCount = Object.values(phraseData).flat().filter(item => item.completed).length;

  const bgColor = isDark ? 'bg-darkbg' : 'bg-secondary';
  const surfaceColor = isDark ? 'bg-darksurface' : 'bg-gray-200';
  const textColor = isDark ? '#F8F8F8' : '#1F2937';
  const subTextColor = isDark ? '#A8A8A8' : '#6B7280';

  return (
    <View className={`flex-1 ${bgColor}`} style={{ paddingTop: insets.top }}>
      <AppHeaderLearn 
        title="Learn Phrases"
        completedCount={completedCount}
        totalCount={allPhrasesCount}
      />

      <ScrollView className="flex-1 p-4" contentContainerStyle={{ paddingBottom: 150 }}>
        
        {/* Category Tabs */}
        <Text style={[styles.sectionTitle, { color: textColor }]}>Choose Category</Text>
        <View className={`flex-row justify-around mb-8 p-1 rounded-full ${surfaceColor}`}>
          {Object.keys(phraseData).map((category) => (
            <TouchableOpacity 
              key={category}
              onPress={() => setActiveCategory(category as CategoryKey)}
              className={`flex-1 items-center rounded-full py-2 ${
                activeCategory === category
                  ? isDark
                    ? 'bg-accent'
                    : 'bg-highlight'
                  : ''
              }`}
            >
              <Text
                style={[
                  styles.categoryText,
                  { color: activeCategory === category ? '#000' : subTextColor },
                ]}
              >
                {category.charAt(0).toUpperCase() + category.slice(1)}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* Phrase List */}
        <Text style={[styles.sectionTitle, { color: textColor }]}>
          {activeCategory.charAt(0).toUpperCase() + activeCategory.slice(1)} Phrases
        </Text>
        <View className="mb-8">
          {phraseData[activeCategory].map((item, index) => (
            <TouchableOpacity 
              key={index}
              onPress={() => setActivePhrase(item.phrase)}
              className={`w-full rounded-lg py-4 mb-2 ${
                activePhrase === item.phrase
                  ? isDark
                    ? 'bg-accent'
                    : 'bg-highlight'
                  : surfaceColor
              }`}
            >
              <View className="flex-row items-center pl-4">
                <Text
                  style={[
                    styles.phraseText,
                    { color: activePhrase === item.phrase ? '#000' : subTextColor },
                  ]}
                >
                  {item.phrase}
                </Text>
              </View>
            </TouchableOpacity>
          ))}
        </View>

        {/* Video Placeholder */}
        <Text style={[styles.sectionTitle, { color: textColor }]}>
          Practice: "{activePhrase}"
        </Text>
        <View
          className={`w-full aspect-video rounded-xl items-center justify-center overflow-hidden mb-6 ${
            isDark ? 'bg-darksurface' : 'bg-gray-800'
          }`}
        >
          <Text style={styles.videoPlaceholder}>Sign for "{activePhrase}"</Text>
        </View>

        {/* Practice Section */}
        <Text style={[styles.sectionTitle, { color: textColor }]}>
          Phrase: {activePhrase}
        </Text>
        <Text style={[styles.descriptionText, { color: subTextColor }]}>
          Practice this common phrase. Pay attention to facial expressions and hand movements for proper communication.
        </Text>

        <View className="flex-row justify-between mb-4">
          {['Slow Motion', 'Repeat', 'Practice'].map((label, i) => (
            <TouchableOpacity
              key={i}
              className={`flex-1 rounded-full py-3 mx-1 items-center border border-accent ${surfaceColor}`}
            >
              <Text style={[styles.buttonText, { color: subTextColor }]}>{label}</Text>
            </TouchableOpacity>
          ))}
        </View>

        <TouchableOpacity className="w-full bg-accent rounded-full py-4 items-center mb-6">
          <Text style={styles.completeButtonText}>Completed</Text>
        </TouchableOpacity>
        
        {/* Phrase Tips */}
        <Text style={[styles.sectionTitle, { color: textColor }]}>Phrase Tips</Text>
        <View className={`p-4 rounded-lg mb-4 ${surfaceColor}`}>
          <View className="flex-row items-center mb-2">
            <MaterialIcons name="emoji-emotions" size={24} color={subTextColor} />
            <Text style={[styles.tipHeader, { color: textColor }]}>Facial Expressions</Text>
          </View>
          <Text style={[styles.tipText, { color: subTextColor }]}>
            Facial expressions are crucial for conveying meaning in sign language.
          </Text>
        </View>
        <View className={`p-4 rounded-lg ${surfaceColor}`}>
          <View className="flex-row items-center mb-2">
            <MaterialIcons name="swap-horiz" size={24} color={subTextColor} />
            <Text style={[styles.tipHeader, { color: textColor }]}>Natural Flow</Text>
          </View>
          <Text style={[styles.tipText, { color: subTextColor }]}>
            Practice smooth transitions between signs for natural communication.
          </Text>
        </View>
      </ScrollView>
    </View>
  );
};

export default Phrases;

const styles = StyleSheet.create({
  sectionTitle: {
    fontFamily: 'Audiowide-Regular',
    fontSize: 18,
    marginBottom: 10,
  },
  categoryText: {
    fontFamily: 'Fredoka-SemiBold',
    fontSize: 16,
  },
  phraseText: {
    fontFamily: 'Fredoka-SemiBold',
    fontSize: 18,
  },
  videoPlaceholder: {
    fontFamily: 'Montserrat-SemiBold',
    fontSize: 16,
    color: '#fff',
  },
  descriptionText: {
    fontFamily: 'Montserrat-SemiBold',
    fontSize: 14,
    marginBottom: 16,
  },
  buttonText: {
    fontFamily: 'Fredoka-SemiBold',
    fontSize: 14,
  },
  completeButtonText: {
    fontFamily: 'Fredoka-SemiBold',
    fontSize: 18,
    color: '#fff',
  },
  tipHeader: {
    fontFamily: 'Fredoka-Regular',
    fontSize: 16,
    marginLeft: 8,
  },
  tipText: {
    fontFamily: 'Montserrat-SemiBold',
    fontSize: 14,
  },
});
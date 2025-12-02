// File: frontend/app/(tabs)/learn/dictionary.tsx
import React, { useState, useMemo } from 'react';
import { 
  View, 
  Text, 
  TextInput, 
  TouchableOpacity, 
  ScrollView, 
  ImageBackground, 
  Dimensions 
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons, MaterialIcons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { useTheme } from '../../../src/ThemeContext';

import { alphabetSigns } from '../../../constants/alphabetSigns';
import { numbersData } from '../../../constants/numbers';
import { phrases } from '../../../constants/phrases';

export default function DictionaryScreen() {
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const { isDark } = useTheme();
  const [searchQuery, setSearchQuery] = useState('');

  const bgColorClass = isDark ? 'bg-darkbg' : 'bg-secondary';
  const textColor = isDark ? 'text-secondary' : 'text-primary';
  const itemBg = isDark ? 'bg-darksurface' : 'bg-white';
  
  // FIX 1: Change border color to highlight
  const borderColor = 'border-highlight'; 

  // Combine all data into one searchable list
  const allItems = useMemo(() => {
    const letters = alphabetSigns.map(l => ({
      id: l.letter,
      title: `${l.letter}`,
      type: 'letter',
      category: 'Alphabet',
      sortKey: l.letter
    }));

    const numbers = numbersData.map(n => ({
      id: n.number.toString(),
      title: `${n.number}`,
      type: 'number',
      category: 'Numbers',
      sortKey: n.number.toString()
    }));

    const phraseItems = phrases.map(p => ({
      id: p.id,
      title: p.text,
      type: 'phrase',
      category: 'Phrase',
      sortKey: p.text
    }));

    return [...letters, ...numbers, ...phraseItems];
  }, []);

  const filteredItems = useMemo(() => {
    if (!searchQuery) return allItems;
    const lower = searchQuery.toLowerCase();
    return allItems.filter(item => 
      item.title.toLowerCase().includes(lower) || 
      item.category.toLowerCase().includes(lower)
    );
  }, [searchQuery, allItems]);

  const handlePressItem = (item: any) => {
    if (item.type === 'letter') {
      router.push({ pathname: '/(tabs)/learn/letters', params: { initialLetter: item.id } });
    } else if (item.type === 'number') {
      router.push({ pathname: '/(tabs)/learn/numbers', params: { initialNumber: item.id } });
    } else if (item.type === 'phrase') {
      router.push({ pathname: '/(tabs)/learn/phrases', params: { initialPhraseId: item.id } });
    }
  };

  // Helper to get icon name based on type
  const getIconName = (type: string) => {
    switch (type) {
      case 'letter': return 'text-fields';
      case 'number': return 'format-list-numbered';
      case 'phrase': return 'record-voice-over';
      default: return 'help-outline';
    }
  };

  return (
    <View className={`flex-1 ${bgColorClass}`}>
      <ImageBackground
        source={require('../../../assets/images/MainBG.png')}
        className="flex-1"
        resizeMode="cover"
      >
        <View className="flex-1" style={{ paddingTop: insets.top }}>
          
          {/* Custom Header */}
          <View>
            <LinearGradient
              colors={['#FF6B00', '#FFAB7B']}
              className="py-3 px-4 flex-row items-center"
            >
              <TouchableOpacity onPress={() => router.back()} className="p-1 absolute left-4 z-10">
                <Ionicons name="arrow-back" size={24} color="black" />
              </TouchableOpacity>
              <View className="flex-1 items-center">
                <Text className="text-primary text-xl font-fredoka-semibold">Dictionary</Text>
                <Text className="text-primary text-xs font-fredoka">Search any sign</Text>
              </View>
            </LinearGradient>
            <LinearGradient
              colors={['rgba(255, 171, 123, 1.0)', 'rgba(255, 171, 123, 0.0)']}
              className="h-4"
            />
          </View>

          {/* Search Bar */}
          <View className="px-4 mt-2 mb-4">
            <View className={`flex-row items-center px-4 py-3 rounded-full border-2 ${isDark ? 'bg-darksurface border-accent' : 'bg-white border-accent'}`}>
              <Ionicons name="search" size={20} color="#FF6B00" style={{ marginRight: 8 }} />
              <TextInput
                placeholder="Search for a sign..."
                placeholderTextColor={isDark ? '#888' : '#999'}
                value={searchQuery}
                onChangeText={setSearchQuery}
                className={`flex-1 font-montserrat-medium text-base ${isDark ? 'text-white' : 'text-black'}`}
              />
              {searchQuery.length > 0 && (
                <TouchableOpacity onPress={() => setSearchQuery('')}>
                  <Ionicons name="close-circle" size={20} color="#FF6B00" />
                </TouchableOpacity>
              )}
            </View>
          </View>

          {/* List */}
          <ScrollView className="flex-1 px-4" contentContainerStyle={{ paddingBottom: 150 }}>
            {filteredItems.map((item, index) => (
              <TouchableOpacity
                key={`${item.type}-${item.id}-${index}`}
                onPress={() => handlePressItem(item)}
                // FIX 1: Applied border-highlight
                className={`mb-3 p-7 rounded-2xl flex-row items-center border ${itemBg} ${borderColor} shadow-sm`}
                activeOpacity={0.7}
              >
                <View className={`w-10 h-10 rounded-full items-center justify-center mr-3 ${isDark ? 'bg-darkhover' : 'bg-gray-100'}`}>
                  {/* FIX 2 & 3: Distinct Icons colored Accent */}
                  <MaterialIcons 
                    name={getIconName(item.type) as any} 
                    size={24} 
                    color="#FF6B00" 
                  />
                </View>
                <View className="flex-1">
                  <Text className={`text-base font-fredoka-medium ${textColor}`}>{item.title}</Text>
                  {/* FIX 4: Label color changed to highlight */}
                  <Text className="text-xs font-montserrat-regular text-highlight uppercase">{item.category}</Text>
                </View>
                {/* FIX 3: Chevron color changed to accent */}
                <MaterialIcons name="chevron-right" size={24} color="#FF6B00" />
              </TouchableOpacity>
            ))}
            {filteredItems.length === 0 && (
              <View className="items-center mt-10">
                <Text className={`font-montserrat-medium ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>No signs found.</Text>
              </View>
            )}
          </ScrollView>

        </View>
      </ImageBackground>
    </View>
  );
}
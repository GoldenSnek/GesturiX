import React, { useState, useCallback } from 'react';
import { 
  View, 
  Text, 
  TouchableOpacity, 
  ScrollView, 
  ImageBackground, 
  ActivityIndicator 
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { MaterialIcons, Ionicons } from '@expo/vector-icons';
import { useRouter, useFocusEffect } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { Video, ResizeMode } from 'expo-av';

import { useTheme } from '../../../src/ThemeContext';
import { getCurrentUserId, getUserSavedItems, unsaveItem, SavedItem } from '../../../utils/supabaseApi';

import { alphabetSigns } from '../../../constants/alphabetSigns';
import { numbersData } from '../../../constants/numbers';
import { phrases } from '../../../constants/phrases';

export default function SavedSignsScreen() {
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const { isDark } = useTheme();
  
  const [savedItems, setSavedItems] = useState<SavedItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [userId, setUserId] = useState<string | null>(null);

  const bgColorClass = isDark ? 'bg-darkbg' : 'bg-secondary';
  const textColor = isDark ? 'text-secondary' : 'text-primary';
  const subTextColor = isDark ? 'text-gray-300' : 'text-gray-600'; 
  const cardBg = isDark ? 'bg-darksurface' : 'bg-white';
  
  const cardBorder = 'border-highlight';

  const loadSavedItems = async () => {
    setLoading(true);
    const uid = await getCurrentUserId();
    setUserId(uid);
    if (uid) {
      const items = await getUserSavedItems(uid);
      items.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
      setSavedItems(items);
    }
    setLoading(false);
  };

  useFocusEffect(
    useCallback(() => {
      loadSavedItems();
    }, [])
  );

  const handleUnsave = async (item: SavedItem) => {
    if (!userId) return;
    
    setSavedItems(prev => prev.filter(i => i.id !== item.id));
    
    await unsaveItem(userId, item.item_type, item.item_identifier);
  };

  const handlePressItem = (item: SavedItem) => {
    if (item.item_type === 'letter') {
      router.push({
        pathname: '/(tabs)/learn/letters',
        params: { initialLetter: item.item_identifier }
      });
    } else if (item.item_type === 'number') {
      router.push({
        pathname: '/(tabs)/learn/numbers',
        params: { initialNumber: item.item_identifier }
      });
    } else if (item.item_type === 'phrase') {
      router.push({
        pathname: '/(tabs)/learn/phrases',
        params: { initialPhraseId: item.item_identifier }
      });
    }
  };

  const getDetailFromId = (item: SavedItem) => {
    if (item.item_type === 'letter') {
      const data = alphabetSigns.find(l => l.letter === item.item_identifier);
      return { 
        title: `Letter ${item.item_identifier}`, 
        subtitle: 'Alphabet', 
        video: data?.image, 
        tips: data?.tips 
      };
    } else if (item.item_type === 'number') {
      const numVal = parseInt(item.item_identifier, 10);
      const data = numbersData.find(n => n.number === numVal);
      return { 
        title: `Number ${item.item_identifier}`, 
        subtitle: 'Numbers', 
        video: data?.video, 
        tips: data?.tips 
      };
    } else if (item.item_type === 'phrase') {
      const data = phrases.find(p => p.id === item.item_identifier);
      return { 
        title: data?.text || 'Unknown Phrase', 
        subtitle: 'Phrase', 
        video: data?.videoUrl, 
        tips: data?.tips 
      };
    }
    return null;
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
                <Text className="text-primary text-xl font-fredoka-semibold">Saved Signs</Text>
                <Text className="text-primary text-xs font-fredoka">Your personal collection</Text>
              </View>
            </LinearGradient>
            <LinearGradient
              colors={['rgba(255, 171, 123, 1.0)', 'rgba(255, 171, 123, 0.0)']}
              className="h-4"
            />
          </View>

          {/* Content */}
          {loading ? (
            <View className="flex-1 justify-center items-center">
              <ActivityIndicator size="large" color="#FF6B00" />
            </View>
          ) : savedItems.length === 0 ? (
            <View className="flex-1 justify-center items-center px-10">
              <MaterialIcons name="bookmark-border" size={64} color={isDark ? '#555' : '#ccc'} />
              <Text className={`text-center mt-4 text-lg font-fredoka-medium ${subTextColor}`}>
                No saved signs yet.
              </Text>
              <Text className={`text-center mt-2 text-sm font-montserrat-regular ${subTextColor}`}>
                Tap the save icon while learning letters, numbers, or phrases to add them here.
              </Text>
            </View>
          ) : (
            <ScrollView className="flex-1 px-4" contentContainerStyle={{ paddingBottom: 100 }}>
              {savedItems.map((item) => {
                const details = getDetailFromId(item);
                if (!details) return null;

                return (
                  <TouchableOpacity 
                    key={item.id} 
                    className={`mb-4 rounded-2xl overflow-hidden border ${cardBorder} ${cardBg} shadow-sm`}
                    activeOpacity={0.9}
                    onPress={() => handlePressItem(item)}
                  >
                    <View className="flex-row p-3">
                      {/* Mini Video Preview */}
                      <View className="w-24 h-24 rounded-xl overflow-hidden bg-black mr-3 relative border border-gray-300">
                        {details.video ? (
                          <Video
                            source={details.video}
                            style={{ width: '100%', height: '100%' }}
                            resizeMode={ResizeMode.COVER}
                            shouldPlay={false}
                            isMuted={true}
                          />
                        ) : (
                          <View className="flex-1 justify-center items-center">
                            <MaterialIcons name="broken-image" size={24} color="white" />
                          </View>
                        )}
                        {/* Play overlay icon */}
                        <View className="absolute inset-0 justify-center items-center bg-black/20">
                           <MaterialIcons name="play-circle-filled" size={32} color="white" />
                        </View>
                      </View>

                      {/* Text Info */}
                      <View className="flex-1 justify-between py-1">
                        <View>
                          <View className="flex-row justify-between items-start">
                            <Text className={`text-lg font-fredoka-semibold ${textColor} flex-1 mr-2`} numberOfLines={1}>
                              {details.title}
                            </Text>
                            <TouchableOpacity 
                              onPress={(e) => {
                                e.stopPropagation();
                                handleUnsave(item);
                              }}
                              className="p-1"
                            >
                              <MaterialIcons name="bookmark" size={24} color="#FF6B00" />
                            </TouchableOpacity>
                          </View>
                          
                          <Text className="text-xs font-montserrat-bold text-highlight uppercase mt-0.5">
                            {details.subtitle}
                          </Text>
                        </View>
                        
                        <Text className={`text-sm font-montserrat-medium ${subTextColor} mt-2`} numberOfLines={2}>
                          Tip: {details.tips}
                        </Text>
                      </View>
                    </View>
                  </TouchableOpacity>
                );
              })}
            </ScrollView>
          )}
        </View>
      </ImageBackground>
    </View>
  );
}
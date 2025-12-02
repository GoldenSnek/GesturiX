// File: frontend/app/(tabs)/learn/videos.tsx
import React from 'react';
import { 
  View, 
  Text, 
  TouchableOpacity, 
  ScrollView, 
  ImageBackground, 
  Image,
  Linking,
  Alert 
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { MaterialIcons, Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { useTheme } from '../../../src/ThemeContext';

// 📺 DATA: Categorized Video List
const videoCategories = [
  {
    categoryTitle: "Fundamentals",
    data: [
      {
        id: '1',
        title: 'ASL Basics | American Sign Language for Beginners',
        description: 'Basic words and alphabet.',
        videoId: '0FcwzMq4iWg', 
        duration: '13:42',
        channel: 'Learn How to Sign'
      },
      {
        id: '3',
        title: 'Introduction to Sign Language Class',
        description: 'Lesson format and basics for beginners.',
        videoId: 'a3GNRqcWYcE',
        duration: '15:30',
        channel: 'ASL Basics'
      },
      {
        id: '4',
        title: 'The Best ASL Course for Beginners | FREE Lesson',
        description: 'Interactive concepts and practical examples.',
        videoId: 'uN6ho9VQARA',
        duration: '35:12',
        channel: 'Learn How to Sign'
      },
      {
        id: '8',
        title: 'Learn Sign Language: Lesson 01 (ASL)',
        description: 'Immersion-based learning with Dr. Bill Vicars.',
        videoId: 'DaMjr4AfYA0',
        duration: '14:20',
        channel: 'Bill Vicars'
      },
    ]
  },
  {
    categoryTitle: "Vocabulary",
    data: [
      {
        id: '2',
        title: '150 Essential ASL Signs | Part 1',
        description: 'Core vocabulary and everyday signs.',
        videoId: '4Ll3OtqAzyw', 
        duration: '23:48',
        channel: 'Learn How to Sign'
      },
      {
        id: '9',
        title: 'Beginner ASL Lesson One Part I: Vocabulary',
        description: 'Focused vocabulary building for beginners.',
        videoId: 'aDoMvhP2u7k',
        duration: '11:18',
        channel: 'Bill Vicars'
      },
      {
        id: '10',
        title: '101 Most Searched ASL Phrases',
        description: 'Quick reference for phrases needed in daily conversation.',
        videoId: 'ChiWKqGbxYc',
        duration: '20:45',
        channel: 'Learn How to Sign'
      },
    ]
  },
  {
    categoryTitle: "Classes & Practice",
    data: [
      {
        id: '7',
        title: 'American Sign Language Class: Part 1',
        description: 'Common phrases and practice.',
        videoId: 'Tq0jj90POWM',
        duration: '32:15',
        channel: 'Bill Vicars'
      },
      {
        id: '6',
        title: 'Intermediate ASL Practice and Review',
        description: 'Moving beyond basics to intermediate skills.',
        videoId: '7jzhDH5KkRM',
        duration: '10:05',
        channel: 'Bill Vicars'
      },
      {
        id: '5',
        title: 'Live ASL Class: Intro to ASL Interpretation',
        description: 'Teacher-led session on interpretation and basics.',
        videoId: 'xGJTj0yjjJI',
        duration: '55:20',
        channel: 'TakeLessons'
      },
    ]
  }
];

export default function VideoLessonsScreen() {
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const { isDark } = useTheme();
  
  const bgColorClass = isDark ? 'bg-darkbg' : 'bg-secondary';
  const textColor = isDark ? 'text-secondary' : 'text-primary';
  const subTextColor = isDark ? 'text-neutral' : 'text-gray-500';
  const cardBg = isDark ? 'bg-darksurface' : 'bg-white';
  
  // FIX 1: Change border color to highlight
  const cardBorder = 'border-highlight';

  const handlePressVideo = async (videoId: string) => {
    const url = `https://www.youtube.com/watch?v=${videoId}`;
    try {
      await Linking.openURL(url);
    } catch (err) {
      console.error("Failed to open video URL:", err);
      Alert.alert("Error", "Could not open the video link. Please check your connection or YouTube app.");
    }
  };

  // FIX 2: Open YouTube search with category pre-filled
  const handleViewMore = async (category: string) => {
    // Prepend "ASL" to ensure relevant results
    const query = encodeURIComponent(`ASL ${category}`);
    const url = `https://www.youtube.com/results?search_query=${query}`;
    
    try {
      await Linking.openURL(url);
    } catch (err) {
      console.error("Failed to open YouTube search:", err);
      Alert.alert("Error", "Could not open YouTube.");
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
                <Text className="text-primary text-xl font-fredoka-semibold">Video Lessons</Text>
                <Text className="text-primary text-xs font-fredoka">Learn ASL through video tutorials</Text>
              </View>
            </LinearGradient>
            <LinearGradient
              colors={['rgba(255, 171, 123, 1.0)', 'rgba(255, 171, 123, 0.0)']}
              className="h-4"
            />
          </View>

          <ScrollView 
            className="flex-1 px-4" 
            contentContainerStyle={{ paddingBottom: 100 }}
          >
            {videoCategories.map((section, sectionIndex) => (
              <View key={sectionIndex} className="mb-6">
                
                {/* Header Row with Title and View More Button */}
                <View className="flex-row justify-between items-center mb-3 mt-4 pr-1">
                  <Text
                    className={`text-lg ${textColor}`}
                    style={{ fontFamily: 'Audiowide-Regular' }}
                  >
                    {section.categoryTitle}
                  </Text>
                  
                  <TouchableOpacity 
                    onPress={() => handleViewMore(section.categoryTitle)}
                    className="flex-row items-center bg-accent/10 px-3 py-1 rounded-full"
                    activeOpacity={0.7}
                  >
                    <Text className="text-xs font-fredoka-medium text-accent mr-1">View More</Text>
                    <Ionicons name="open-outline" size={12} color="#FF6B00" />
                  </TouchableOpacity>
                </View>

                {section.data.map((item) => (
                  <TouchableOpacity 
                    key={item.id} 
                    // FIX 1: Applied border-highlight here
                    className={`mb-4 rounded-2xl overflow-hidden border ${cardBorder} ${cardBg} shadow-sm`}
                    activeOpacity={0.9}
                    onPress={() => handlePressVideo(item.videoId)}
                  >
                    <View className="flex-row p-3">
                      {/* Thumbnail Container */}
                      <View className="w-32 h-24 rounded-xl overflow-hidden bg-black mr-3 relative border border-gray-300 items-center justify-center">
                        <Image
                          source={{ uri: `https://img.youtube.com/vi/${item.videoId}/mqdefault.jpg` }}
                          style={{ width: '100%', height: '100%' }}
                          resizeMode="cover"
                        />
                        {/* Play Icon Overlay */}
                        <View className="absolute inset-0 justify-center items-center bg-black/30">
                           <MaterialIcons name="play-circle-fill" size={32} color="white" />
                        </View>
                      </View>

                      {/* Text Info */}
                      <View className="flex-1 justify-between py-1">
                        <View>
                          <Text className={`text-base font-fredoka-semibold ${textColor} mb-1`} numberOfLines={2}>
                            {item.title}
                          </Text>
                          <Text className="text-xs font-fredoka text-accent uppercase">
                            {item.channel}
                          </Text>
                        </View>
                        
                        <View className="flex-row items-center">
                            <MaterialIcons name="access-time" size={12} color={isDark ? '#A8A8A8' : '#666'} />
                            <Text className={`text-xs font-montserrat-medium ml-1 ${subTextColor}`}>
                                {item.duration}
                            </Text>
                        </View>
                        
                        <Text className={`text-xs font-montserrat-regular ${subTextColor} mt-1`} numberOfLines={1}>
                          {item.description}
                        </Text>
                      </View>
                    </View>
                  </TouchableOpacity>
                ))}
              </View>
            ))}
          </ScrollView>
        </View>
      </ImageBackground>
    </View>
  );
}
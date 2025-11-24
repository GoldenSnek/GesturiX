// File: frontend/app/(tabs)/learn/leaderboard.tsx
import React, { useState, useCallback } from 'react';
import { 
  View, 
  Text, 
  TouchableOpacity, 
  FlatList, 
  ImageBackground, 
  Image,
  ActivityIndicator 
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons, MaterialIcons } from '@expo/vector-icons';
import { useRouter, useFocusEffect } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { useTheme } from '../../../src/ThemeContext';
import { fetchLeaderboard } from '../../../utils/supabaseApi';
import { supabase } from '../../../src/supabaseClient';


interface LeaderboardUser {
  lessons_completed: number;
  days_streak: number;
  practice_hours: number;
  profiles: {
    username: string;
    photo_url?: string | null;
  };
}


export default function LeaderboardScreen() {
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const { isDark } = useTheme();
  const [users, setUsers] = useState<LeaderboardUser[]>([]);
  const [loading, setLoading] = useState(true);


  const bgColorClass = isDark ? 'bg-darkbg' : 'bg-secondary';
  const textColor = isDark ? 'text-secondary' : 'text-primary';
  const itemBg = isDark ? 'bg-darksurface' : 'bg-white';
  const borderColor = isDark ? 'border-gray-700' : 'border-gray-200';


  const loadData = async () => {
    setLoading(true);
    const data = await fetchLeaderboard();
    setUsers(data as unknown as LeaderboardUser[]);
    setLoading(false);
  };


  useFocusEffect(
    useCallback(() => {
      loadData();
    }, [])
  );


  const getAvatarUrl = (path: string | null | undefined) => {
    if (!path) return 'https://ui-avatars.com/api/?name=User&background=random';
    const { data } = supabase.storage.from('avatars').getPublicUrl(path);
    return data.publicUrl;
  };


  const renderItem = ({ item, index }: { item: LeaderboardUser; index: number }) => {
    let rankColor = '#999';
    let rankSize = 16;
    if (index === 0) { rankColor = '#FFD700'; rankSize = 24; } // Gold
    else if (index === 1) { rankColor = '#C0C0C0'; rankSize = 22; } // Silver
    else if (index === 2) { rankColor = '#CD7F32'; rankSize = 20; } // Bronze


    return (
      <View 
        className={`flex-row items-center p-4 mb-3 rounded-2xl border ${borderColor} ${itemBg} shadow-sm`}
      >
        <View className="w-8 items-center justify-center mr-2">
          {index < 3 ? (
            <MaterialIcons name="emoji-events" size={rankSize} color={rankColor} />
          ) : (
            <Text className={`font-audiowide text-lg text-gray-400`}>{index + 1}</Text>
          )}
        </View>


        <Image
          source={{ uri: getAvatarUrl(item.profiles?.photo_url) }}
          className="w-12 h-12 rounded-full mr-3 border border-gray-300"
        />


        <View className="flex-1">
          <Text className={`font-fredoka-semibold text-lg ${textColor}`} numberOfLines={1}>
            {item.profiles?.username || 'Unknown'}
          </Text>
          <View className="flex-row items-center">
             <MaterialIcons name="local-fire-department" size={14} color="#FF6B00" />
             <Text className="text-xs text-gray-500 font-montserrat-medium ml-1">
                {item.days_streak} day streak
             </Text>
          </View>
        </View>


        <View className="items-end">
          <Text className="text-accent font-audiowide text-xl">{item.lessons_completed}</Text>
          <Text className="text-xs text-gray-400 font-fredoka">Lessons</Text>
        </View>
      </View>
    );
  };


  return (
    <View className={`flex-1 ${bgColorClass}`}>
      <ImageBackground
        source={require('../../../assets/images/MainBG.png')}
        className="flex-1"
        resizeMode="cover"
      >
        <View className="flex-1" style={{ paddingTop: insets.top }}>
          
          {/* Custom Header - Centered with larger text */}
          <View>
            <LinearGradient
              colors={['#FF6B00', '#FFAB7B']}
              className="py-3 px-4 flex-row items-center"
            >
              <TouchableOpacity onPress={() => router.back()} className="p-1 absolute left-4 z-10">
                <Ionicons name="arrow-back" size={24} color="black" />
              </TouchableOpacity>
              <View className="flex-1 items-center">
                <Text className="text-primary text-xl font-fredoka-semibold">Leaderboard</Text>
                <Text className="text-primary text-xs font-fredoka">Top learners this week</Text>
              </View>
            </LinearGradient>
            <LinearGradient
              colors={['rgba(255, 171, 123, 1.0)', 'rgba(255, 171, 123, 0.0)']}
              className="h-4"
            />
          </View>


          {loading ? (
            <View className="flex-1 justify-center items-center">
              <ActivityIndicator size="large" color="#FF6B00" />
            </View>
          ) : (
            <FlatList
              data={users}
              keyExtractor={(item, index) => index.toString()}
              renderItem={renderItem}
              contentContainerStyle={{ padding: 16, paddingBottom: 100 }}
              ListEmptyComponent={
                <Text className={`text-center mt-10 font-montserrat-regular ${textColor}`}>
                  No data available yet.
                </Text>
              }
            />
          )}
        </View>
      </ImageBackground>
    </View>
  );
}

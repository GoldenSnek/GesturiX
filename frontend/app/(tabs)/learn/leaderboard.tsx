// File: frontend/app/(tabs)/learn/leaderboard.tsx
import React, { useState, useCallback } from 'react';
import { 
  View, 
  Text, 
  TouchableOpacity, // <--- Added back
  FlatList, 
  ImageBackground, 
  Image,
  ActivityIndicator,
  RefreshControl
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';
import { useRouter, useFocusEffect } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { useTheme } from '../../../src/ThemeContext';
import { fetchLeaderboard, getCurrentUserId } from '../../../utils/supabaseApi';
import { supabase } from '../../../src/supabaseClient';

interface LeaderboardUser {
  user_id: string;
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
  const [refreshing, setRefreshing] = useState(false);
  const [currentUserId, setCurrentUserId] = useState<string | null>(null);

  const bgColorClass = isDark ? 'bg-darkbg' : 'bg-secondary';
  const textColor = isDark ? 'text-secondary' : 'text-primary';
  const itemBg = isDark ? 'bg-darksurface' : 'bg-white';
  const borderColor = isDark ? 'border-gray-700' : 'border-gray-200';

  const loadData = async () => {
    if (!refreshing) setLoading(true);
    
    try {
      const [leaderboardData, uid] = await Promise.all([
        fetchLeaderboard(),
        getCurrentUserId()
      ]);

      setUsers(leaderboardData as unknown as LeaderboardUser[]);
      setCurrentUserId(uid);
    } catch (error) {
      console.error("Error loading leaderboard data:", error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    loadData();
  }, []);

  useFocusEffect(
    useCallback(() => {
      loadData();
    }, [])
  );

  const getAvatarUrl = (path: string | null | undefined) => {
    if (!path) return 'https://ui-avatars.com/api/?name=User&background=random';
    if (path.startsWith('http')) return path;
    const { data } = supabase.storage.from('avatars').getPublicUrl(path);
    return data.publicUrl;
  };

  const renderItem = ({ item, index }: { item: LeaderboardUser; index: number }) => {
    const isCurrentUser = item.user_id === currentUserId;

    // Define dynamic styles for the current user
    const containerBorder = isCurrentUser ? 'border-accent' : borderColor;
    const containerBg = isCurrentUser 
      ? (isDark ? 'bg-orange-900/20' : 'bg-orange-50') // Subtle highlight background
      : itemBg;
    const containerBorderWidth = isCurrentUser ? 'border-2' : 'border';

    let rankColor = '#999';
    let rankSize = 16;
    if (index === 0) { rankColor = '#FFD700'; rankSize = 24; }
    else if (index === 1) { rankColor = '#C0C0C0'; rankSize = 22; }
    else if (index === 2) { rankColor = '#CD7F32'; rankSize = 20; }

    return (
      <View 
        className={`flex-row items-center p-4 mb-3 rounded-2xl ${containerBorderWidth} ${containerBorder} ${containerBg} shadow-sm`}
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
          className={`w-12 h-12 rounded-full mr-3 border ${isCurrentUser ? 'border-accent' : 'border-gray-300'}`}
        />

        <View className="flex-1">
          <Text className={`font-fredoka-semibold text-lg ${textColor}`} numberOfLines={1}>
            {item.profiles?.username || 'Unknown'} 
            {isCurrentUser && <Text className="text-accent text-sm"> (You)</Text>}
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
          
          {/* Custom Header */}
          <View>
            <LinearGradient
              colors={['#FF6B00', '#FFAB7B']}
              className="py-3 px-4 flex-row items-center"
            >
              <TouchableOpacity onPress={() => router.back()} className="p-1 absolute left-4 z-10">
                <MaterialIcons name="arrow-back" size={24} color="black" />
              </TouchableOpacity>
              <View className="flex-1 items-center">
                <Text className="text-primary text-xl font-fredoka-semibold">Leaderboard</Text>
                <Text className="text-primary text-xs font-fredoka">Community Rankings</Text>
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
              refreshControl={
                <RefreshControl
                  refreshing={refreshing}
                  onRefresh={onRefresh}
                  colors={['#FF6B00']}
                  tintColor={'#FF6B00'}
                />
              }
              ListEmptyComponent={
                <Text className={`text-center mt-10 font-montserrat-regular ${textColor}`}>
                  No users found.
                </Text>
              }
            />
          )}
        </View>
      </ImageBackground>
    </View>
  );
}
// File: frontend/app/(tabs)/learn/leaderboard.tsx
import React, { useState, useCallback, useEffect } from 'react';
import { 
  View, 
  Text, 
  TouchableOpacity, 
  FlatList, 
  ImageBackground, 
  Image,
  ActivityIndicator,
  RefreshControl,
  Modal,
  Dimensions,
  Animated,
  StyleSheet
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { MaterialIcons, Ionicons } from '@expo/vector-icons';
import { useRouter, useFocusEffect } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { useTheme } from '../../../src/ThemeContext';
import { 
  fetchLeaderboard, 
  getCurrentUserId, 
  getProfileLikeCount, 
  getHasUserLiked, 
  likeProfile, 
  unlikeProfile 
} from '../../../utils/supabaseApi';
import { supabase } from '../../../src/supabaseClient';
import * as Haptics from 'expo-haptics';


const { height: SCREEN_HEIGHT } = Dimensions.get('window');


interface LeaderboardUser {
  user_id: string;
  lessons_completed: number;
  days_streak: number;
  practice_hours: number;
  profiles: {
    username: string;
    photo_url?: string | null;
    created_at?: string;
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


  // Modal State
  const [selectedUser, setSelectedUser] = useState<LeaderboardUser | null>(null);
  const [modalVisible, setModalVisible] = useState(false);
  const [profileLikes, setProfileLikes] = useState(0);
  const [isLiked, setIsLiked] = useState(false);
  const [likeLoading, setLikeLoading] = useState(false);
  const [slideAnim] = useState(new Animated.Value(SCREEN_HEIGHT));
  const [fadeAnim] = useState(new Animated.Value(0));


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


  // Modal Animation
  useEffect(() => {
    if (modalVisible) {
      Animated.parallel([
        Animated.spring(slideAnim, {
          toValue: 0,
          useNativeDriver: true,
          tension: 65,
          friction: 11
        }),
        Animated.timing(fadeAnim, {
          toValue: 1,
          duration: 200,
          useNativeDriver: true
        })
      ]).start();
    } else {
      Animated.parallel([
        Animated.timing(slideAnim, {
          toValue: SCREEN_HEIGHT,
          duration: 250,
          useNativeDriver: true
        }),
        Animated.timing(fadeAnim, {
          toValue: 0,
          duration: 200,
          useNativeDriver: true
        })
      ]).start();
    }
  }, [modalVisible]);


  // Like Logic
  useEffect(() => {
    let isMounted = true;
    const fetchLikeStatus = async () => {
      if (selectedUser && currentUserId && modalVisible) {
        setLikeLoading(true);
        const [count, userLiked] = await Promise.all([
          getProfileLikeCount(selectedUser.user_id),
          getHasUserLiked(currentUserId, selectedUser.user_id)
        ]);
        if (isMounted) {
          setProfileLikes(count);
          setIsLiked(userLiked);
          setLikeLoading(false);
        }
      }
    };
    fetchLikeStatus();
    return () => { isMounted = false; };
  }, [selectedUser, currentUserId, modalVisible]);


  const handleToggleLike = async () => {
    if (!selectedUser || !currentUserId || likeLoading) return;


    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);


    // Optimistic Update
    const previousLiked = isLiked;
    const previousCount = profileLikes;
    
    setIsLiked(!previousLiked);
    setProfileLikes(previousLiked ? previousCount - 1 : previousCount + 1);


    try {
      if (previousLiked) {
        await unlikeProfile(currentUserId, selectedUser.user_id);
      } else {
        await likeProfile(currentUserId, selectedUser.user_id);
      }
    } catch (error) {
      // Revert on error
      setIsLiked(previousLiked);
      setProfileLikes(previousCount);
    }
  };


  const getAvatarUrl = (path: string | null | undefined) => {
    if (!path) return 'https://ui-avatars.com/api/?name=User&background=random';
    if (path.startsWith('http')) return path;
    const { data } = supabase.storage.from('avatars').getPublicUrl(path);
    return data.publicUrl;
  };


  const openProfile = (user: LeaderboardUser) => {
    setSelectedUser(user);
    setModalVisible(true);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  };


  const closeModal = () => {
    setModalVisible(false);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  };


  const getRankSuffix = (index: number) => {
    const num = index + 1;
    if (num === 1) return 'st';
    if (num === 2) return 'nd';
    if (num === 3) return 'rd';
    return 'th';
  };


  const renderItem = ({ item, index }: { item: LeaderboardUser; index: number }) => {
    const isCurrentUser = item.user_id === currentUserId;


    const containerBorder = isCurrentUser ? 'border-accent' : borderColor;
    const containerBg = isCurrentUser 
      ? (isDark ? 'bg-orange-900/20' : 'bg-orange-50') 
      : itemBg;
    const containerBorderWidth = isCurrentUser ? 'border-2' : 'border';


    let rankColor = '#999';
    let rankSize = 16;
    if (index === 0) { rankColor = '#FFD700'; rankSize = 24; }
    else if (index === 1) { rankColor = '#C0C0C0'; rankSize = 22; }
    else if (index === 2) { rankColor = '#CD7F32'; rankSize = 20; }


    return (
      <TouchableOpacity 
        onPress={() => openProfile(item)}
        activeOpacity={0.9}
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
      </TouchableOpacity>
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


          {/* Professional Profile Modal */}
          <Modal
            animationType="none"
            transparent={true}
            visible={modalVisible}
            onRequestClose={closeModal}
          >
            <View style={styles.modalOverlay}>
              {/* Backdrop */}
              <Animated.View 
                style={[
                  styles.backdrop,
                  { opacity: fadeAnim }
                ]}
              >
                <TouchableOpacity 
                  style={StyleSheet.absoluteFill} 
                  activeOpacity={1} 
                  onPress={closeModal} 
                />
              </Animated.View>
              
              {/* Modal Content */}
              <Animated.View 
                style={[
                  styles.modalContainer,
                  {
                    transform: [{ translateY: slideAnim }]
                  }
                ]}
              >
                <View 
                  className={`rounded-t-3xl overflow-hidden ${isDark ? 'bg-[#1E1E1E]' : 'bg-white'}`}
                  style={styles.modalContent}
                >
                  {/* Top Gradient Accent */}
                  <LinearGradient
                    colors={['#FF6B00', '#FFAB7B']}
                    className="h-24"
                  />
                  
                  {/* Drag Handle */}
                  <View className="absolute top-2 self-center w-12 h-1.5 bg-white/40 rounded-full" />


                  {/* Close Button */}
                  <TouchableOpacity 
                    className="absolute top-3 right-4 w-10 h-10 rounded-full bg-white/20 items-center justify-center z-10"
                    onPress={closeModal}
                    activeOpacity={0.7}
                  >
                    <Ionicons name="close" size={24} color="white" />
                  </TouchableOpacity>


                  {selectedUser && (
                    <View className="px-6 pb-8" style={{ marginTop: -50 }}>
                      {/* Avatar with Shadow & Rank Badge */}
                      <View className="items-center mb-4">
                        <View style={styles.avatarContainer}>
                          <Image
                            source={{ uri: getAvatarUrl(selectedUser.profiles?.photo_url) }}
                            className="w-32 h-32 rounded-full border-4 border-white"
                          />
                          {/* Rank Badge */}
                          <View
                            style={{
                              position: 'absolute',
                              bottom: -8,
                              right: -8,
                              width: 40,
                              height: 40,
                              borderRadius: 20,
                              borderWidth: 4,
                              borderColor: '#fff',
                              shadowColor: '#000',
                              shadowOffset: { width: 0, height: 4 },
                              shadowOpacity: 0.3,
                              shadowRadius: 8,
                              elevation: 6,
                              justifyContent: 'center',
                              alignItems: 'center',
                              overflow: 'hidden',
                            }}
                          >
                            <LinearGradient
                              colors={['#FF6B00', '#FF8C42']}
                              style={{ width: 40, height: 40, borderRadius: 20, justifyContent: 'center', alignItems: 'center' }}
                            >
                              <MaterialIcons name="emoji-events" size={22} color="white" />
                            </LinearGradient>
                          </View>
                        </View>
                      </View>


                      {/* Name & Rank */}
                      <View className="items-center mb-2">
                        <Text className={`text-2xl font-fredoka-bold text-center ${textColor}`}>
                          {selectedUser.profiles?.username}
                        </Text>
                        <View className="flex-row items-center mt-1">
                          <View className={`px-3 py-1 rounded-full ${isDark ? 'bg-accent/20' : 'bg-orange-100'}`}>
                            <Text className="text-accent font-audiowide text-sm">
                              {users.findIndex(u => u.user_id === selectedUser.user_id) + 1}
                              {getRankSuffix(users.findIndex(u => u.user_id === selectedUser.user_id))} Place
                            </Text>
                          </View>
                        </View>
                      </View>


                      {/* Member Since */}
                      <View className="items-center mb-6">
                        <View className="flex-row items-center">
                          <Ionicons name="calendar-outline" size={14} color="#999" />
                          <Text className="text-xs font-montserrat-medium text-gray-500 ml-1.5">
                            Member since {new Date(selectedUser.profiles?.created_at || Date.now()).toLocaleDateString(undefined, { 
                              year: 'numeric', 
                              month: 'short'
                            })}
                          </Text>
                        </View>
                      </View>


                      {/* Stats Grid with Enhanced Design */}
                      <View className="mb-6">
                        <Text className={`text-sm font-fredoka-semibold mb-3 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                          Performance Stats
                        </Text>
                        <View className="flex-row justify-between">
                          {/* Lessons */}
                          <View className={`flex-1 items-center p-4 rounded-2xl mx-1 ${isDark ? 'bg-darksurface border border-gray-700' : 'bg-gray-50'}`}>
                            <View className="w-12 h-12 rounded-full bg-accent/10 items-center justify-center mb-2">
                              <MaterialIcons name="check-circle" size={26} color="#FF6B00" />
                            </View>
                            <Text className={`text-xl font-fredoka-bold ${textColor}`}>
                              {selectedUser.lessons_completed}
                            </Text>
                            <Text className="text-xs text-gray-500 font-montserrat mt-0.5">Lessons</Text>
                          </View>
                          
                          {/* Streak */}
                          <View className={`flex-1 items-center p-4 rounded-2xl mx-1 ${isDark ? 'bg-darksurface border border-gray-700' : 'bg-gray-50'}`}>
                            <View className="w-12 h-12 rounded-full bg-accent/10 items-center justify-center mb-2">
                              <MaterialIcons name="local-fire-department" size={26} color="#FF6B00" />
                            </View>
                            <Text className={`text-xl font-fredoka-bold ${textColor}`}>
                              {selectedUser.days_streak}
                            </Text>
                            <Text className="text-xs text-gray-500 font-montserrat mt-0.5">Day Streak</Text>
                          </View>


                          {/* Hours */}
                          <View className={`flex-1 items-center p-4 rounded-2xl mx-1 ${isDark ? 'bg-darksurface border border-gray-700' : 'bg-gray-50'}`}>
                            <View className="w-12 h-12 rounded-full bg-accent/10 items-center justify-center mb-2">
                              <MaterialIcons name="timer" size={26} color="#FF6B00" />
                            </View>
                            <Text className={`text-xl font-fredoka-bold ${textColor}`}>
                              {selectedUser.practice_hours.toFixed(1)}
                            </Text>
                            <Text className="text-xs text-gray-500 font-montserrat mt-0.5">Hours</Text>
                          </View>
                        </View>
                      </View>


                      {/* Divider */}
                      <View className={`h-px mb-6 ${isDark ? 'bg-gray-700' : 'bg-gray-200'}`} />


                      {/* Like Section */}
                      <View>
                        <Text className={`text-sm font-fredoka-semibold mb-3 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                          Show Appreciation
                        </Text>
                        <TouchableOpacity
                          onPress={handleToggleLike}
                          activeOpacity={0.8}
                          disabled={likeLoading}
                          className="overflow-hidden rounded-2xl shadow-md"
                        >
                          {isLiked ? (
                            <LinearGradient
                              colors={['#EF4444', '#DC2626']}
                              start={{ x: 0, y: 0 }}
                              end={{ x: 1, y: 1 }}
                              className="flex-row items-center justify-center py-4 px-6"
                            >
                              <Ionicons name="heart" size={26} color="white" />
                              <View className="flex-1 ml-3">
                                <Text className="text-lg font-fredoka-bold text-white">
                                  You liked this profile
                                </Text>
                                <Text className="text-xs font-montserrat text-white/80">
                                  {profileLikes} {profileLikes === 1 ? 'person likes' : 'people like'} this profile
                                </Text>
                              </View>
                              {profileLikes > 0 && (
                                <View className="px-3 py-1.5 rounded-full bg-white/20">
                                  <Text className="text-sm font-audiowide text-white">
                                    {profileLikes}
                                  </Text>
                                </View>
                              )}
                            </LinearGradient>
                          ) : (
                            <View className={`flex-row items-center justify-center py-4 px-6 ${isDark ? 'bg-darksurface border border-gray-700' : 'bg-gray-50'}`}>
                              <Ionicons name="heart-outline" size={26} color="#FF6B00" />
                              <View className="flex-1 ml-3">
                                <Text className={`text-lg font-fredoka-bold ${textColor}`}>
                                  Give them a like!
                                </Text>
                                <Text className="text-xs font-montserrat text-gray-500">
                                  {profileLikes} {profileLikes === 1 ? 'person likes' : 'people like'} this profile
                                </Text>
                              </View>
                              {profileLikes > 0 && (
                                <View className="px-3 py-1.5 rounded-full bg-accent/10">
                                  <Text className="text-sm font-audiowide text-accent">
                                    {profileLikes}
                                  </Text>
                                </View>
                              )}
                            </View>
                          )}
                        </TouchableOpacity>
                      </View>


                    </View>
                  )}
                </View>
              </Animated.View>
            </View>
          </Modal>


        </View>
      </ImageBackground>
    </View>
  );
}


const styles = StyleSheet.create({
  modalOverlay: {
    flex: 1,
    justifyContent: 'flex-end',
  },
  backdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
  },
  modalContainer: {
    maxHeight: '90%',
  },
  modalContent: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -4 },
    shadowOpacity: 0.3,
    shadowRadius: 12,
    elevation: 20,
  },
  avatarContainer: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.3,
    shadowRadius: 16,
    elevation: 12,
  },
});

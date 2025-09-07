const supabase = require('../config/supabase');
const { calculateTax } = require('../utils/taxCalculator');

class PaymentService {
  async processDailyPayouts() {
    try {
      // Get yesterday's revenue
      const yesterday = new Date();
      yesterday.setDate(yesterday.getDate() - 1);
      const dateStr = yesterday.toISOString().split('T')[0];
      
      const { data: revenueData, error } = await supabase
        .from('daily_revenue')
        .select('*')
        .eq('date', dateStr)
        .single();
      
      if (error) throw error;
      
      if (revenueData) {
        const totalRevenue = revenueData.ad_revenue + revenueData.iap_revenue;
        const ownerPayout = totalRevenue * 0.6; // 60% to owner
        const aiFund = totalRevenue * 0.4; // 40% to AI fund
        
        // Calculate tax
        const tax = calculateTax(ownerPayout);
        
        // Record payout
        const { error: payoutError } = await supabase
          .from('payouts')
          .insert({
            date: dateStr,
            total_revenue: totalRevenue,
            owner_payout: ownerPayout,
            ai_fund: aiFund,
            tax_estimated: tax,
            status: 'processed'
          });
        
        if (payoutError) throw payoutError;
        
        // TODO: Actually transfer funds to owner's account
        console.log(`Processed payout: Owner: R${ownerPayout}, AI Fund: R${aiFund}, Tax: R${tax}`);
        
        return { success: true, ownerPayout, aiFund, tax };
      }
      
      return { success: false, message: 'No revenue data for yesterday' };
    } catch (error) {
      console.error('Error processing payouts:', error);
      throw error;
    }
  }
  
  async getFinancialDashboardData() {
    try {
      // Get last 30 days of revenue
      const thirtyDaysAgo = new Date();
      thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
      
      const { data: revenueData, error } = await supabase
        .from('daily_revenue')
        .select('*')
        .gte('date', thirtyDaysAgo.toISOString().split('T')[0])
        .order('date', { ascending: true });
      
      if (error) throw error;
      
      // Get player geographic distribution
      const { data: playerData } = await supabase
        .from('players')
        .select('country, city');
      
      // Process geographic data
      const geoData = this.processGeoData(playerData);
      
      return {
        revenueData,
        geoData
      };
    } catch (error) {
      console.error('Error getting financial data:', error);
      throw error;
    }
  }
  
  processGeoData(playerData) {
    const countries = {};
    const cities = {};
    
    playerData.forEach(player => {
      if (player.country) {
        countries[player.country] = (countries[player.country] || 0) + 1;
      }
      
      if (player.city && player.country) {
        const key = `${player.country}-${player.city}`;
        cities[key] = (cities[key] || 0) + 1;
      }
    });
    
    return { countries, cities };
  }
}

module.exports = new PaymentService();
